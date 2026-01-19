# SPDX-License-Identifier: GPL-2.0
obj-$(CONFIG_F2FS_FS) += f2fs.o

f2fs-y		:= dir.o file.o inode.o namei.o hash.o super.o inline.o
f2fs-y		+= checkpoint.o gc.o data.o node.o segment.o recovery.o
f2fs-y		+= shrinker.o extent_cache.o sysfs.o
f2fs-y		+= filemgr.o kernelMLP.o mlp_config_stage1_haccu.o mlp_config_stage1_hrecall.o mlp_config_stage2.o
f2fs-$(CONFIG_F2FS_STAT_FS) += debug.o
f2fs-$(CONFIG_F2FS_FS_XATTR) += xattr.o
f2fs-$(CONFIG_F2FS_FS_POSIX_ACL) += acl.o
f2fs-$(CONFIG_FS_VERITY) += verity.o
f2fs-$(CONFIG_F2FS_FS_COMPRESSION) += compress.o
f2fs-$(CONFIG_F2FS_IOSTAT) += iostat.o
