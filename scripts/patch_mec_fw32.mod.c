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
	{ 0x6360be9f, "dma_free_attrs" },
	{ 0xdcf837ae, "pci_iounmap" },
	{ 0x19bb1bcc, "pci_dev_put" },
	{ 0xcbae5412, "__const_udelay" },
	{ 0x8ac9537f, "pci_get_device" },
	{ 0x1a925fa4, "pci_iomap" },
	{ 0x3cf6abed, "dma_alloc_attrs" },
	{ 0x9b5acfd3, "request_firmware" },
	{ 0xa53f4e29, "memcpy" },
	{ 0x1abc7887, "release_firmware" },
	{ 0xbd03ed67, "vmemmap_base" },
	{ 0xbd03ed67, "page_offset_base" },
	{ 0x1c489eb6, "register_kprobe" },
	{ 0x90a48d82, "__ubsan_handle_out_of_bounds" },
	{ 0xd272d446, "__stack_chk_fail" },
	{ 0x635ab929, "param_ops_int" },
	{ 0xd272d446, "__fentry__" },
	{ 0xd272d446, "__x86_return_thunk" },
	{ 0xe8213e80, "_printk" },
	{ 0x7a8e92c6, "unregister_kprobe" },
	{ 0xd268ca91, "module_layout" },
};

static const u32 ____version_ext_crcs[]
__used __section("__version_ext_crcs") = {
	0x6360be9f,
	0xdcf837ae,
	0x19bb1bcc,
	0xcbae5412,
	0x8ac9537f,
	0x1a925fa4,
	0x3cf6abed,
	0x9b5acfd3,
	0xa53f4e29,
	0x1abc7887,
	0xbd03ed67,
	0xbd03ed67,
	0x1c489eb6,
	0x90a48d82,
	0xd272d446,
	0x635ab929,
	0xd272d446,
	0xd272d446,
	0xe8213e80,
	0x7a8e92c6,
	0xd268ca91,
};
static const char ____version_ext_names[]
__used __section("__version_ext_names") =
	"dma_free_attrs\0"
	"pci_iounmap\0"
	"pci_dev_put\0"
	"__const_udelay\0"
	"pci_get_device\0"
	"pci_iomap\0"
	"dma_alloc_attrs\0"
	"request_firmware\0"
	"memcpy\0"
	"release_firmware\0"
	"vmemmap_base\0"
	"page_offset_base\0"
	"register_kprobe\0"
	"__ubsan_handle_out_of_bounds\0"
	"__stack_chk_fail\0"
	"param_ops_int\0"
	"__fentry__\0"
	"__x86_return_thunk\0"
	"_printk\0"
	"unregister_kprobe\0"
	"module_layout\0"
;

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "EE52BDCC8C07C4BDFC353A7");
