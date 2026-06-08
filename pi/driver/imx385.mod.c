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
	{ 0x5aee23f5, "i2c_register_driver" },
	{ 0xdd70a476, "regmap_write" },
	{ 0xa7889b3, "_dev_info" },
	{ 0x1d9f777c, "_dev_err" },
	{ 0xb6e6d99d, "clk_disable" },
	{ 0xb077e70a, "clk_unprepare" },
	{ 0x1715a9ea, "regulator_bulk_disable" },
	{ 0x4dfa8d4b, "mutex_lock" },
	{ 0x3213f038, "mutex_unlock" },
	{ 0xfd9ee7fe, "__v4l2_subdev_state_get_crop" },
	{ 0x89fe0b01, "__v4l2_subdev_state_get_format" },
	{ 0x2a2ad949, "i2c_del_driver" },
	{ 0xc3055d20, "usleep_range_state" },
	{ 0x7c9a7371, "clk_prepare" },
	{ 0x815588a6, "clk_enable" },
	{ 0x90f098be, "regulator_bulk_enable" },
	{ 0x33297427, "v4l2_async_unregister_subdev" },
	{ 0x263f3602, "v4l2_ctrl_handler_free" },
	{ 0x2587b615, "__pm_runtime_disable" },
	{ 0x437d0342, "__pm_runtime_set_status" },
	{ 0xe2822320, "__v4l2_find_nearest_size" },
	{ 0x3ac81b38, "__v4l2_ctrl_s_ctrl" },
	{ 0x316650e2, "__v4l2_ctrl_s_ctrl_int64" },
	{ 0xdcb764ad, "memset" },
	{ 0xf0fdf6cb, "__stack_chk_fail" },
	{ 0x36a78de3, "devm_kmalloc" },
	{ 0xd852811d, "v4l2_subdev_init" },
	{ 0x929c4eb7, "__devm_regmap_init_i2c" },
	{ 0x6d64f6ec, "__dev_fwnode" },
	{ 0x7e30b1de, "fwnode_graph_get_next_endpoint" },
	{ 0xc95ea0f9, "v4l2_fwnode_endpoint_alloc_parse" },
	{ 0xe856a90, "devm_clk_get" },
	{ 0xa6527791, "fwnode_property_read_u32_array" },
	{ 0x76d9b876, "clk_set_rate" },
	{ 0x273de036, "devm_regulator_bulk_get" },
	{ 0xcefb0c9f, "__mutex_init" },
	{ 0x9aaf43cd, "v4l2_ctrl_handler_init_class" },
	{ 0x75c4b665, "v4l2_ctrl_new_std" },
	{ 0x9e781eac, "v4l2_ctrl_new_int_menu" },
	{ 0x133d5f87, "v4l2_ctrl_new_std_menu_items" },
	{ 0xa8eb0c77, "v4l2_i2c_subdev_init" },
	{ 0xa3732c8f, "media_entity_pads_init" },
	{ 0x247cff22, "__v4l2_subdev_init_finalize" },
	{ 0x8444a114, "__v4l2_async_register_subdev" },
	{ 0x4dcc3608, "pm_runtime_enable" },
	{ 0xfda399a4, "__pm_runtime_idle" },
	{ 0xce5bb82a, "v4l2_fwnode_endpoint_free" },
	{ 0x5d975c3c, "__pm_runtime_resume" },
	{ 0xa65c6def, "alt_cb_patch_nops" },
	{ 0xf9a482f9, "msleep" },
	{ 0xc38b7ac0, "v4l2_ctrl_handler_setup" },
	{ 0xd1b9afa6, "pm_runtime_get_if_in_use" },
	{ 0xa2eb131b, "__v4l2_ctrl_modify_range" },
	{ 0x580fd1bb, "v4l2_subdev_link_validate" },
	{ 0x474e54d2, "module_layout" },
};

MODULE_INFO(depends, "videodev,v4l2-async,regmap-i2c,v4l2-fwnode,mc");

MODULE_ALIAS("of:N*T*Csony,imx385");
MODULE_ALIAS("of:N*T*Csony,imx385C*");

MODULE_INFO(srcversion, "E0C4B8862FC77412D6B00F1");
