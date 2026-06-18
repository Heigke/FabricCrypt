savedcmd_fw17b_diag.mod := printf '%s\n'   fw17b_diag.o | awk '!x[$$0]++ { print("./"$$0) }' > fw17b_diag.mod
