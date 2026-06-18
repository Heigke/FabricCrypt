#include <linux/module.h>
#include <linux/export-internal.h>
#include <linux/compiler.h>

MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};



static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x90a48d82, "__ubsan_handle_out_of_bounds" },
	{ 0xb069d2a9, "iommu_get_domain_for_dev" },
	{ 0xdb3b699f, "memremap" },
	{ 0x3d77375e, "iommu_iova_to_phys" },
	{ 0xa21e2f95, "memunmap" },
	{ 0x97dd6ca9, "ioremap_wc" },
	{ 0x12ad300e, "iounmap" },
	{ 0x82fd7238, "__ubsan_handle_shift_out_of_bounds" },
	{ 0x3cf6abed, "dma_alloc_attrs" },
	{ 0x97dd6ca9, "ioremap" },
	{ 0xbd03ed67, "page_offset_base" },
	{ 0x9b5acfd3, "request_firmware_direct" },
	{ 0x1abc7887, "release_firmware" },
	{ 0xa53f4e29, "memcpy" },
	{ 0x5a844b26, "__x86_indirect_thunk_r13" },
	{ 0xc46670f1, "slow_virt_to_phys" },
	{ 0x8ac9537f, "pci_get_device" },
	{ 0x1a925fa4, "pci_iomap" },
	{ 0x9b5acfd3, "request_firmware" },
	{ 0x27683a56, "memset" },
	{ 0x635ab929, "param_ops_int" },
	{ 0xd272d446, "__fentry__" },
	{ 0xd272d446, "__x86_return_thunk" },
	{ 0xe8213e80, "_printk" },
	{ 0x5a844b26, "__x86_indirect_thunk_r15" },
	{ 0x1c489eb6, "register_kprobe" },
	{ 0x7a8e92c6, "unregister_kprobe" },
	{ 0xd272d446, "__stack_chk_fail" },
	{ 0xdcf837ae, "pci_iounmap" },
	{ 0x19bb1bcc, "pci_dev_put" },
	{ 0x6360be9f, "dma_free_attrs" },
	{ 0x1b60315e, "copy_from_kernel_nofault" },
	{ 0x5a844b26, "__x86_indirect_thunk_r14" },
	{ 0xa59dd599, "pci_find_ext_capability" },
	{ 0x22029f10, "pci_read_config_dword" },
	{ 0x5a844b26, "__x86_indirect_thunk_r12" },
	{ 0xcbae5412, "__const_udelay" },
	{ 0x1b60315e, "copy_to_kernel_nofault" },
	{ 0x5a844b26, "__x86_indirect_thunk_rbx" },
	{ 0x5a844b26, "__x86_indirect_thunk_rax" },
	{ 0xf5bae445, "__virt_addr_valid" },
	{ 0xd268ca91, "module_layout" },
};

static const u32 ____version_ext_crcs[]
__used __section("__version_ext_crcs") = {
	0x90a48d82,
	0xb069d2a9,
	0xdb3b699f,
	0x3d77375e,
	0xa21e2f95,
	0x97dd6ca9,
	0x12ad300e,
	0x82fd7238,
	0x3cf6abed,
	0x97dd6ca9,
	0xbd03ed67,
	0x9b5acfd3,
	0x1abc7887,
	0xa53f4e29,
	0x5a844b26,
	0xc46670f1,
	0x8ac9537f,
	0x1a925fa4,
	0x9b5acfd3,
	0x27683a56,
	0x635ab929,
	0xd272d446,
	0xd272d446,
	0xe8213e80,
	0x5a844b26,
	0x1c489eb6,
	0x7a8e92c6,
	0xd272d446,
	0xdcf837ae,
	0x19bb1bcc,
	0x6360be9f,
	0x1b60315e,
	0x5a844b26,
	0xa59dd599,
	0x22029f10,
	0x5a844b26,
	0xcbae5412,
	0x1b60315e,
	0x5a844b26,
	0x5a844b26,
	0xf5bae445,
	0xd268ca91,
};
static const char ____version_ext_names[]
__used __section("__version_ext_names") =
	"__ubsan_handle_out_of_bounds\0"
	"iommu_get_domain_for_dev\0"
	"memremap\0"
	"iommu_iova_to_phys\0"
	"memunmap\0"
	"ioremap_wc\0"
	"iounmap\0"
	"__ubsan_handle_shift_out_of_bounds\0"
	"dma_alloc_attrs\0"
	"ioremap\0"
	"page_offset_base\0"
	"request_firmware_direct\0"
	"release_firmware\0"
	"memcpy\0"
	"__x86_indirect_thunk_r13\0"
	"slow_virt_to_phys\0"
	"pci_get_device\0"
	"pci_iomap\0"
	"request_firmware\0"
	"memset\0"
	"param_ops_int\0"
	"__fentry__\0"
	"__x86_return_thunk\0"
	"_printk\0"
	"__x86_indirect_thunk_r15\0"
	"register_kprobe\0"
	"unregister_kprobe\0"
	"__stack_chk_fail\0"
	"pci_iounmap\0"
	"pci_dev_put\0"
	"dma_free_attrs\0"
	"copy_from_kernel_nofault\0"
	"__x86_indirect_thunk_r14\0"
	"pci_find_ext_capability\0"
	"pci_read_config_dword\0"
	"__x86_indirect_thunk_r12\0"
	"__const_udelay\0"
	"copy_to_kernel_nofault\0"
	"__x86_indirect_thunk_rbx\0"
	"__x86_indirect_thunk_rax\0"
	"__virt_addr_valid\0"
	"module_layout\0"
;

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "01EE2634C074416C54D8F3F");
