savedcmd_patch_mec_fw35.mod := printf '%s\n'   patch_mec_fw35.o | awk '!x[$$0]++ { print("./"$$0) }' > patch_mec_fw35.mod
