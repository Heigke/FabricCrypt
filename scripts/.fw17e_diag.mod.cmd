savedcmd_fw17e_diag.mod := printf '%s\n'   fw17e_diag.o | awk '!x[$$0]++ { print("./"$$0) }' > fw17e_diag.mod
