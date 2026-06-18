savedcmd_fw17f_diag.mod := printf '%s\n'   fw17f_diag.o | awk '!x[$$0]++ { print("./"$$0) }' > fw17f_diag.mod
