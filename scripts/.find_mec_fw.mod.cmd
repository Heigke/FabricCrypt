savedcmd_find_mec_fw.mod := printf '%s\n'   find_mec_fw.o | awk '!x[$$0]++ { print("./"$$0) }' > find_mec_fw.mod
