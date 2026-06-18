savedcmd_patch_mec_fw9.mod := printf '%s\n'   patch_mec_fw9.o | awk '!x[$$0]++ { print("./"$$0) }' > patch_mec_fw9.mod
