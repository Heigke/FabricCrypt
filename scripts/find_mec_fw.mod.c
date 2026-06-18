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
	{ 0xd272d446, "__fentry__" },
	{ 0xaed2e23c, "node_states" },
	{ 0x494c552b, "_find_first_bit" },
	{ 0xb50ec83b, "node_data" },
	{ 0x86632fd6, "_find_next_bit" },
	{ 0xe8213e80, "_printk" },
	{ 0xb1ad3f2f, "boot_cpu_data" },
	{ 0x211f9d4e, "mem_section" },
	{ 0xb51225d6, "pcpu_hot" },
	{ 0xbd03ed67, "vmemmap_base" },
	{ 0xbd03ed67, "page_offset_base" },
	{ 0xd272d446, "__SCT__preempt_schedule" },
	{ 0xd272d446, "__x86_return_thunk" },
	{ 0xd268ca91, "module_layout" },
};

static const u32 ____version_ext_crcs[]
__used __section("__version_ext_crcs") = {
	0xd272d446,
	0xaed2e23c,
	0x494c552b,
	0xb50ec83b,
	0x86632fd6,
	0xe8213e80,
	0xb1ad3f2f,
	0x211f9d4e,
	0xb51225d6,
	0xbd03ed67,
	0xbd03ed67,
	0xd272d446,
	0xd272d446,
	0xd268ca91,
};
static const char ____version_ext_names[]
__used __section("__version_ext_names") =
	"__fentry__\0"
	"node_states\0"
	"_find_first_bit\0"
	"node_data\0"
	"_find_next_bit\0"
	"_printk\0"
	"boot_cpu_data\0"
	"mem_section\0"
	"pcpu_hot\0"
	"vmemmap_base\0"
	"page_offset_base\0"
	"__SCT__preempt_schedule\0"
	"__x86_return_thunk\0"
	"module_layout\0"
;

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "70D8BE20AAB5B45FD2D3DC3");
