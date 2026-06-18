/*
 * probe_rebar.c — Resize BAR probe + TMR physical access test
 *
 * mode=1: Verify aper_base by reading known VRAM BO via derived phys addr
 * mode=2: Attempt runtime BAR resize to 8GB and access TMR
 * mode=3: TOCTOU — patch GART BO + trigger GPU reset + race to patch TMR
 * mode=4: PSP TA fuzzing — send crafted commands via RAS TA invoke
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/kprobes.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("FEEL");
MODULE_DESCRIPTION("Resize BAR + TMR access probe");

static int mode = 1;
module_param(mode, int, 0644);

/* AMD GPU PCI IDs */
#define AMD_VENDOR 0x1002
#define GPU_DEVICE 0x1586

/* Forward declarations for persistent TMR monitor */
static struct timer_list tmr_timer;
static void __iomem *tmr_persistent_map;
static int tmr_poll_count;
static int tmr_found;
static void tmr_poll_callback(struct timer_list *t);

static void probe_phys_mapping(struct pci_dev *pdev)
{
	resource_size_t bar0_start = pci_resource_start(pdev, 0);
	resource_size_t bar0_len   = pci_resource_len(pdev, 0);
	void __iomem *map;

	pr_info("rebar: BAR0 start=0x%llX len=0x%llX (%llu MB)\n",
		(u64)bar0_start, (u64)bar0_len, (u64)bar0_len >> 20);

	/* Test 1: Read VRAM at BAR0 + 0 via ioremap */
	map = ioremap_wc(bar0_start, 0x1000);
	if (map) {
		u32 v = readl(map);
		pr_info("rebar: BAR0[0] via ioremap = 0x%08X\n", v);
		iounmap(map);
	} else {
		pr_info("rebar: BAR0[0] ioremap failed\n");
	}

	/* Test 2: Read VRAM at offset 0x7EE000 (known trampoline BO) */
	if (0x7EE000 < bar0_len) {
		map = ioremap_wc(bar0_start + 0x7EE000, 0x1000);
		if (map) {
			u32 v[4];
			int i;
			for (i = 0; i < 4; i++)
				v[i] = readl(map + i * 4);
			pr_info("rebar: VRAM+0x7EE000: %08X %08X %08X %08X\n",
				v[0], v[1], v[2], v[3]);
			iounmap(map);
		} else {
			pr_info("rebar: VRAM+0x7EE000 ioremap failed\n");
		}
	}

	/* Test 3: Find adev via amdgpu driver, get TMR kptr, derive aper_base */
	{
		/* Search for adev by looking at BAR0 region in driver */
		struct amdgpu_device_stub {
			char pad[0x10]; /* drm_device * */
		};

		/* Use kallsyms to find amdgpu_device_rreg — from there we can
		 * find adev. But simpler: use the known working approach from
		 * our kernel modules — find psp symbol then derive adev. */

		void *adev = NULL;
		{
			/* Find adev via PCI driver data → drm_dev → adev */
			void *drm_dev = pci_get_drvdata(pdev);
			if (drm_dev) {
				/* ddev is at offset 0x10 in amdgpu_device */
				adev = (void *)((u8 *)drm_dev - 0x10);
				pr_info("rebar: drm_dev=0x%px adev=0x%px\n",
					drm_dev, adev);
			}
		}

		if (adev) {
			/* Read TMR kptr and mc from known offsets */
			u64 val0 = *(u64 *)((u8 *)adev + 0x3B920);
			u64 val1 = *(u64 *)((u8 *)adev + 0x3B928);
			u64 tmr_kptr, tmr_mc;

			if ((val0 >> 48) == 0xFFFF) {
				tmr_kptr = val0; tmr_mc = val1;
			} else if ((val1 >> 48) == 0xFFFF) {
				tmr_kptr = val1; tmr_mc = val0;
			} else {
				tmr_kptr = 0; tmr_mc = 0;
			}

			pr_info("rebar: TMR kptr=0x%llX mc=0x%llX\n", tmr_kptr, tmr_mc);

			if (tmr_kptr) {
				phys_addr_t tmr_phys = slow_virt_to_phys(
					(void *)(unsigned long)tmr_kptr);
				u64 fb_base = 0x8000000000ULL;
				u64 mc_off = tmr_mc - fb_base;
				u64 aper_base;

				pr_info("rebar: TMR phys = 0x%llX\n", (u64)tmr_phys);

				if (tmr_phys && tmr_phys != ~0ULL) {
					aper_base = tmr_phys - mc_off;
					pr_info("rebar: aper_base = 0x%llX\n", aper_base);

					/* Verify: read trampoline BO via aper_base */
					{
						u64 tramp_phys = aper_base + 0x7EE000;
						void __iomem *tm;
						pr_info("rebar: Trampoline phys = 0x%llX\n",
							tramp_phys);

						tm = ioremap_wc(tramp_phys, 0x1000);
						if (tm) {
							u32 v[4];
							int i;
							for (i = 0; i < 4; i++)
								v[i] = readl(tm + i * 4);
							pr_info("rebar: Trampoline via aper: %08X %08X %08X %08X\n",
								v[0], v[1], v[2], v[3]);
							iounmap(tm);
						} else {
							pr_info("rebar: Trampoline aper ioremap failed\n");
						}
					}

					/* Now try TMR BO */
					{
						u64 tmr_bo_mc = 0x97E0000000ULL;
						u64 tmr_bo_phys = aper_base +
							(tmr_bo_mc - fb_base);
						void __iomem *tm;

						pr_info("rebar: TMR BO phys = 0x%llX\n",
							tmr_bo_phys);

						tm = ioremap_wc(tmr_bo_phys, 0x1000);
						if (tm) {
							u32 v = readl(tm);
							pr_info("rebar: TMR BO[0] = 0x%08X%s\n",
								v, v == 0xFFFFFFFF ?
								" (PROTECTED)" : " (READABLE!)");

							/* If readable, scan for firmware! */
							if (v != 0xFFFFFFFF) {
								pr_info("rebar: *** TMR BO ACCESSIBLE! ***\n");
								/* Scan for 0x04070663 pattern */
							}
							iounmap(tm);
						} else {
							pr_info("rebar: TMR BO ioremap failed\n");
						}
					}

					/* Also try: BAR0_start vs aper_base comparison */
					pr_info("rebar: BAR0_start=0x%llX aper_base=0x%llX diff=0x%llX\n",
						(u64)bar0_start, aper_base,
						(u64)bar0_start - aper_base);
				}
			}
		}
	}
}

static void attempt_bar_resize(struct pci_dev *pdev)
{
	int ret;
	resource_size_t bar0_start = pci_resource_start(pdev, 0);
	resource_size_t bar0_len = pci_resource_len(pdev, 0);

	pr_info("rebar: === BAR RESIZE ATTEMPT ===\n");
	pr_info("rebar: Current BAR0: start=0x%llX len=0x%llX (%llu MB)\n",
		(u64)bar0_start, (u64)bar0_len, (u64)bar0_len >> 20);

	/* Read Resize BAR capability */
	{
		u32 rebar_cap, rebar_ctrl;
		int pos = pci_find_ext_capability(pdev, PCI_EXT_CAP_ID_REBAR);

		if (!pos) {
			pr_info("rebar: Resize BAR capability not found!\n");
			return;
		}
		pr_info("rebar: Resize BAR at config offset 0x%X\n", pos);

		pci_read_config_dword(pdev, pos + 4, &rebar_cap);
		pci_read_config_dword(pdev, pos + 8, &rebar_ctrl);
		pr_info("rebar: BAR0 capability = 0x%08X\n", rebar_cap);
		pr_info("rebar: BAR0 control    = 0x%08X (size=%d → %llu MB)\n",
			rebar_ctrl, (rebar_ctrl >> 8) & 0x1F,
			1ULL << (((rebar_ctrl >> 8) & 0x1F) + 20 - 20));

		/* Check supported sizes */
		{
			int bit;
			pr_info("rebar: Supported sizes:");
			for (bit = 0; bit < 32; bit++) {
				if (rebar_cap & (1 << bit))
					pr_cont(" %lluMB", (1ULL << bit));
			}
			pr_cont("\n");
		}

		/* Attempt resize to 8GB (size=13) */
		{
			u32 new_ctrl;
			u16 cmd;

			pr_info("rebar: Attempting resize to 8GB...\n");

			/* Step 1: Disable memory decode */
			pci_read_config_word(pdev, PCI_COMMAND, &cmd);
			pr_info("rebar: PCI_COMMAND = 0x%04X\n", cmd);

			/* WARNING: Disabling memory decode will kill display on iGPU!
			 * Only proceed if we can re-enable quickly. */

			/* Check if the kernel has pci_resize_resource */
			ret = pci_resize_resource(pdev, 0, 33); /* 2^33 = 8GB */
			if (ret == 0) {
				pr_info("rebar: *** BAR RESIZED TO 8GB! ***\n");
				bar0_start = pci_resource_start(pdev, 0);
				bar0_len = pci_resource_len(pdev, 0);
				pr_info("rebar: New BAR0: start=0x%llX len=0x%llX (%llu MB)\n",
					(u64)bar0_start, (u64)bar0_len,
					(u64)bar0_len >> 20);
			} else {
				pr_info("rebar: pci_resize_resource failed: %d\n", ret);
				pr_info("rebar: Trying manual resize via config space...\n");

				/* Manual approach: write directly to resize BAR control */
				new_ctrl = (rebar_ctrl & ~0x1F00) | (13 << 8); /* size=13 → 8GB */
				pr_info("rebar: Writing control 0x%08X (was 0x%08X)\n",
					new_ctrl, rebar_ctrl);

				/* DANGEROUS: Disable memory decode first */
				pci_write_config_word(pdev, PCI_COMMAND, cmd & ~PCI_COMMAND_MEMORY);
				udelay(100);

				pci_write_config_dword(pdev, pos + 8, new_ctrl);
				udelay(100);

				/* Read back */
				pci_read_config_dword(pdev, pos + 8, &rebar_ctrl);
				pr_info("rebar: Control readback = 0x%08X (size=%d)\n",
					rebar_ctrl, (rebar_ctrl >> 8) & 0x1F);

				/* Re-enable memory decode */
				pci_write_config_word(pdev, PCI_COMMAND, cmd);
				udelay(100);

				if (((rebar_ctrl >> 8) & 0x1F) == 13) {
					pr_info("rebar: *** BAR RESIZE SUCCEEDED! ***\n");
					/* Now update BAR address for 8GB alignment */
				} else {
					pr_info("rebar: BAR resize rejected by hardware\n");
				}
			}
		}
	}
}

static int __init probe_rebar_init(void)
{
	struct pci_dev *pdev;

	pr_info("rebar: === RESIZE BAR PROBE mode=%d ===\n", mode);

	pdev = pci_get_device(AMD_VENDOR, GPU_DEVICE, NULL);
	if (!pdev) {
		pr_info("rebar: GPU not found, trying any AMD device...\n");
		pdev = pci_get_device(AMD_VENDOR, PCI_ANY_ID, NULL);
	}
	if (!pdev) {
		pr_info("rebar: No AMD GPU found\n");
		return -ENODEV;
	}

	pr_info("rebar: Found %04X:%04X at %s\n",
		pdev->vendor, pdev->device, pci_name(pdev));

	switch (mode) {
	case 1:
		probe_phys_mapping(pdev);
		break;
	case 2:
		attempt_bar_resize(pdev);
		break;
	default:
		pr_info("rebar: Unknown mode %d\n", mode);
	}

	/* Mode 3: TOCTOU — Monitor TMR during GPU reset */
	if (mode == 3) {
		resource_size_t bar0_start = pci_resource_start(pdev, 0);
		void *drm_dev = pci_get_drvdata(pdev);
		void *adev = drm_dev ? (void *)((u8 *)drm_dev - 0x10) : NULL;

		pr_info("rebar: === TOCTOU TMR RACE ===\n");

		if (adev) {
			u64 val0 = *(u64 *)((u8 *)adev + 0x3B920);
			u64 val1 = *(u64 *)((u8 *)adev + 0x3B928);
			u64 tmr_kptr = (val0 >> 48) == 0xFFFF ? val0 : val1;
			u64 tmr_mc = (val0 >> 48) == 0xFFFF ? val1 : val0;
			phys_addr_t tmr_phys = slow_virt_to_phys(
				(void *)(unsigned long)tmr_kptr);
			u64 aper_base = tmr_phys - (tmr_mc - 0x8000000000ULL);
			u64 tmr_bo_phys = aper_base + (0x97E0000000ULL - 0x8000000000ULL);
			void __iomem *tmr_map;
			void __iomem *fw_map = NULL;
			int scan_pages = 2048; /* Scan 8MB of TMR BO */
			u64 fw_found_off = 0;

			pr_info("rebar: aper_base=0x%llX TMR_BO_phys=0x%llX\n",
				aper_base, tmr_bo_phys);

			/* Pre-map the TMR BO region */
			tmr_map = ioremap_wc(tmr_bo_phys, 0x800000); /* 8MB */
			if (!tmr_map) {
				pr_info("rebar: TMR BO ioremap failed\n");
				goto toctou_done;
			}

			/* Also map the first 8MB of VRAM for live-monitoring
			 * the firmware reload */
			fw_map = ioremap_wc(aper_base, 0x800000);

			/* Phase 1: Pre-check — TMR should be all 0xFF */
			{
				u32 v = readl(tmr_map);
				pr_info("rebar: TMR BO pre-check: 0x%08X\n", v);
			}

			/* Phase 2: Continuously monitor TMR while we trigger
			 * GPU reset via debugfs. The monitoring loop runs
			 * in the init function, and we trigger reset from
			 * userspace BEFORE loading this module. That won't work.
			 *
			 * Alternative: We write to the gpu_recover debugfs
			 * from kernel space using filp_open. */
			pr_info("rebar: Triggering GPU reset...\n");
			{
				/* Open and write to gpu_recover debugfs */
				struct file *fp;
				loff_t pos = 0;
				char buf[] = "1\n";
				int i, found = 0;
				u32 last_val = 0xFFFFFFFF;

				/* Start monitoring TMR in a tight loop BEFORE reset */
				pr_info("rebar: Starting TMR monitor + reset...\n");

				/* Trigger reset */
				fp = filp_open("/sys/kernel/debug/dri/0000:c3:00.0/amdgpu_gpu_recover",
					       O_WRONLY, 0);
				if (IS_ERR(fp)) {
					pr_info("rebar: Cannot open gpu_recover: %ld\n",
						PTR_ERR(fp));

					/* Monitor TMR anyway for 2 seconds */
					for (i = 0; i < 200000 && !found; i++) {
						u32 v = readl(tmr_map);
						if (v != 0xFFFFFFFF && v != last_val) {
							pr_info("rebar: *** TMR CHANGED at i=%d: 0x%08X ***\n",
								i, v);
							last_val = v;
							found = 1;

							/* Quick scan for firmware */
							{
								int j;
								for (j = 0; j < 256; j++) {
									u32 w = readl(tmr_map + j * 4);
									if (w == 0x04070663) {
										pr_info("rebar: *** FW at TMR+%d! ***\n",
											j * 4);
									}
								}
							}
						}
						if (i % 50000 == 0)
							pr_info("rebar: Monitor i=%d val=0x%08X\n",
								i, readl(tmr_map));
					}
				} else {
					/* Write reset trigger while monitoring */
					pr_info("rebar: gpu_recover opened, writing...\n");

					/* Monitor in parallel with reset
					 * (simplified: alternating poll + write) */
					kernel_write(fp, buf, 2, &pos);
					pr_info("rebar: Reset triggered, monitoring TMR...\n");

					for (i = 0; i < 1000000 && !found; i++) {
						u32 v = readl(tmr_map);
						if (v != 0xFFFFFFFF) {
							pr_info("rebar: *** TMR READABLE at i=%d: 0x%08X ***\n",
								i, v);
							last_val = v;
							found = 1;

							/* Dump first 64 words */
							{
								int j;
								for (j = 0; j < 64; j++) {
									u32 w = readl(tmr_map + j * 4);
									if (j % 8 == 0)
										pr_info("rebar: TMR[%03X]:", j*4);
									pr_cont(" %08X", w);
									if (j % 8 == 7)
										pr_cont("\n");
								}
							}

							/* Search for firmware in first 8MB */
							{
								u64 off;
								for (off = 0; off < 0x800000 - 8; off += 4) {
									u32 w = readl(tmr_map + off);
									if (w == 0x04070663) {
										u32 w1 = readl(tmr_map + off + 4);
										if (w1 == 0x00060663) {
											pr_info("rebar: *** FW CODE at TMR_BO+0x%llX ***\n",
												off);
											fw_found_off = off;
											break;
										}
									}
								}
							}
						}

						/* Also check if VRAM firmware BO changed */
						if (fw_map && (i % 100000 == 0)) {
							u32 fv = readl(fw_map + 0x7EE000);
							pr_info("rebar: VRAM[0x7EE000] = 0x%08X @ i=%d\n",
								fv, i);
						}
					}

					if (!found)
						pr_info("rebar: TMR stayed protected during reset\n");

					filp_close(fp, NULL);
				}
			}

			if (fw_map) iounmap(fw_map);
			iounmap(tmr_map);
		}
toctou_done:;
	}

	/* Mode 4: Scan PSP C2PMSG mailbox during/after reset for state info */
	if (mode == 4) {
		void *drm_dev = pci_get_drvdata(pdev);
		void *adev = drm_dev ? (void *)((u8 *)drm_dev - 0x10) : NULL;

		pr_info("rebar: === PSP STATE PROBE ===\n");

		if (adev) {
			/* Read PSP C2PMSG registers (safe to READ, never WRITE!) */
			u32 gc_base0 = 0; /* need to find */
			int i;

			/* Find gc_base by scanning known register signature */
			for (i = 0; i < 0x1000; i += 4) {
				u32 *p = (u32 *)((u8 *)adev + i);
				u32 v;
				if (copy_from_kernel_nofault(&v, p, 4))
					continue;
				/* Look for values around 0x2800 (known gc_base offset range) */
				if (v >= 0x2400 && v <= 0x3000 && (v & 0xFF) == 0) {
					pr_info("rebar: Candidate gc_base0 at adev+0x%X = 0x%X\n",
						i, v);
				}
			}

			/* Read PSP ring status from adev.psp fields */
			pr_info("rebar: Scanning adev for PSP context...\n");
			{
				/* PSP context has fence_buf, cmd_buf, tmr_bo etc.
				 * Look for known PSP MC addresses near TMR BO */
				int off;
				for (off = 0; off < 0x10000; off += 8) {
					u64 val = *(u64 *)((u8 *)adev + off);
					if (val == 0x97E0000000ULL) {
						pr_info("rebar: TMR BO MC at adev+0x%X\n", off);
					}
					/* PSP fence buffer MC address */
					if (val >= 0x80000000ULL && val <= 0x80100000ULL &&
					    (val & 0xFFF) == 0) {
						pr_info("rebar: PSP buf at adev+0x%X: mc=0x%llX\n",
							off, val);
					}
				}
			}
		}
	}

	/* Mode 5: Persistent TMR monitor with 1ms timer */
	if (mode == 5) {
		void *drm_dev = pci_get_drvdata(pdev);
		void *adev = drm_dev ? (void *)((u8 *)drm_dev - 0x10) : NULL;

		pr_info("rebar: === PERSISTENT TMR MONITOR ===\n");

		if (adev) {
			u64 val0 = *(u64 *)((u8 *)adev + 0x3B920);
			u64 val1 = *(u64 *)((u8 *)adev + 0x3B928);
			u64 tmr_kptr = (val0 >> 48) == 0xFFFF ? val0 : val1;
			u64 tmr_mc = (val0 >> 48) == 0xFFFF ? val1 : val0;
			phys_addr_t tmr_phys = slow_virt_to_phys(
				(void *)(unsigned long)tmr_kptr);
			u64 aper_base = tmr_phys - (tmr_mc - 0x8000000000ULL);
			u64 tmr_bo_phys = aper_base + (0x97E0000000ULL - 0x8000000000ULL);

			pr_info("rebar: TMR BO phys = 0x%llX\n", tmr_bo_phys);

			tmr_persistent_map = ioremap_wc(tmr_bo_phys, 0x800000);
			if (tmr_persistent_map) {
				pr_info("rebar: TMR mapped. Starting 1ms timer poll.\n");
				pr_info("rebar: NOW trigger GPU reset with:\n");
				pr_info("rebar:   echo 1 > /sys/kernel/debug/dri/0000:c3:00.0/amdgpu_gpu_recover\n");

				tmr_poll_count = 0;
				tmr_found = 0;
				timer_setup(&tmr_timer, tmr_poll_callback, 0);
				mod_timer(&tmr_timer, jiffies + msecs_to_jiffies(1));

				pci_dev_put(pdev);
				return 0; /* Stay loaded! */
			} else {
				pr_info("rebar: TMR ioremap failed\n");
			}
		}
	}

	/* Mode 6: Attempt BAR resize to 8GB */
	if (mode == 6) {
		attempt_bar_resize(pdev);
	}

	pci_dev_put(pdev);
	return 0;
}

/* Mode 5: Persistent TMR monitor — stays loaded, polls TMR every 10ms
 * via timer. When TMR becomes readable, dumps content.
 * Designed to catch TOCTOU window during external GPU reset. */
static struct timer_list tmr_timer;
static void __iomem *tmr_persistent_map;
static int tmr_poll_count;
static int tmr_found;

static void tmr_poll_callback(struct timer_list *t)
{
	if (tmr_persistent_map && !tmr_found) {
		u32 v = readl(tmr_persistent_map);
		tmr_poll_count++;

		if (v != 0xFFFFFFFF) {
			pr_info("rebar: *** TMR READABLE! poll=%d val=0x%08X ***\n",
				tmr_poll_count, v);
			tmr_found = 1;

			/* Dump first 256 bytes */
			{
				int i;
				for (i = 0; i < 64; i++) {
					u32 w = readl(tmr_persistent_map + i * 4);
					if (i % 8 == 0)
						pr_info("rebar: [%03X]:", i * 4);
					pr_cont(" %08X", w);
					if (i % 8 == 7)
						pr_cont("\n");
				}
			}

			/* Search for firmware pattern */
			{
				u64 off;
				for (off = 0; off < 0x800000 - 8; off += 4) {
					u32 w = readl(tmr_persistent_map + off);
					if (w == 0x04070663) {
						u32 w1 = readl(tmr_persistent_map + off + 4);
						if (w1 == 0x00060663)
							pr_info("rebar: *** FW at TMR+0x%llX ***\n", off);
					}
				}
			}
		} else if (tmr_poll_count % 100 == 0) {
			pr_info("rebar: TMR poll %d: still protected\n", tmr_poll_count);
		}

		if (!tmr_found && tmr_poll_count < 10000)
			mod_timer(&tmr_timer, jiffies + msecs_to_jiffies(1));
	}
}

static void __exit probe_rebar_exit(void)
{
	if (tmr_persistent_map) {
		del_timer_sync(&tmr_timer);
		iounmap(tmr_persistent_map);
		tmr_persistent_map = NULL;
	}
	pr_info("rebar: unloaded (polls=%d found=%d)\n", tmr_poll_count, tmr_found);
}

module_init(probe_rebar_init);
module_exit(probe_rebar_exit);
