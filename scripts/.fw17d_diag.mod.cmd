savedcmd_fw17d_diag.mod := printf '%s\n'   fw17d_diag.o | awk '!x[$$0]++ { print("./"$$0) }' > fw17d_diag.mod
