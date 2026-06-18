savedcmd_patch_mec_fw13.mod := printf '%s\n'   patch_mec_fw13.o | awk '!x[$$0]++ { print("./"$$0) }' > patch_mec_fw13.mod
