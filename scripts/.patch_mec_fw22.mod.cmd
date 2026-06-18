savedcmd_patch_mec_fw22.mod := printf '%s\n'   patch_mec_fw22.o | awk '!x[$$0]++ { print("./"$$0) }' > patch_mec_fw22.mod
