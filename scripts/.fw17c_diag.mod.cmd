savedcmd_fw17c_diag.mod := printf '%s\n'   fw17c_diag.o | awk '!x[$$0]++ { print("./"$$0) }' > fw17c_diag.mod
