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
	{ 0x8ac9537f, "pci_get_device" },
	{ 0xbd03ed67, "random_kmalloc_seed" },
	{ 0xcbaa3a2c, "kmalloc_caches" },
	{ 0xfb794c10, "__kmalloc_cache_noprof" },
	{ 0xcb8b6ec6, "kfree" },
	{ 0x19bb1bcc, "pci_dev_put" },
	{ 0x5a844b26, "__x86_indirect_thunk_rax" },
	{ 0x1b60315e, "copy_from_kernel_nofault" },
	{ 0x90a48d82, "__ubsan_handle_out_of_bounds" },
	{ 0xd272d446, "__stack_chk_fail" },
	{ 0x635ab929, "param_ops_int" },
	{ 0xe8213e80, "_printk" },
	{ 0xd272d446, "__x86_return_thunk" },
	{ 0xd268ca91, "module_layout" },
};

static const u32 ____version_ext_crcs[]
__used __section("__version_ext_crcs") = {
	0xd272d446,
	0x8ac9537f,
	0xbd03ed67,
	0xcbaa3a2c,
	0xfb794c10,
	0xcb8b6ec6,
	0x19bb1bcc,
	0x5a844b26,
	0x1b60315e,
	0x90a48d82,
	0xd272d446,
	0x635ab929,
	0xe8213e80,
	0xd272d446,
	0xd268ca91,
};
static const char ____version_ext_names[]
__used __section("__version_ext_names") =
	"__fentry__\0"
	"pci_get_device\0"
	"random_kmalloc_seed\0"
	"kmalloc_caches\0"
	"__kmalloc_cache_noprof\0"
	"kfree\0"
	"pci_dev_put\0"
	"__x86_indirect_thunk_rax\0"
	"copy_from_kernel_nofault\0"
	"__ubsan_handle_out_of_bounds\0"
	"__stack_chk_fail\0"
	"param_ops_int\0"
	"_printk\0"
	"__x86_return_thunk\0"
	"module_layout\0"
;

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "5F0467495E5BA8AB43DB2AA");
