savedcmd_patch_mec_fw.mod := printf '%s\n'   patch_mec_fw.o | awk '!x[$$0]++ { print("./"$$0) }' > patch_mec_fw.mod
