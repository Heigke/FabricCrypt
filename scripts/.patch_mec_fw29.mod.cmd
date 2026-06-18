savedcmd_patch_mec_fw29.mod := printf '%s\n'   patch_mec_fw29.o | awk '!x[$$0]++ { print("./"$$0) }' > patch_mec_fw29.mod
