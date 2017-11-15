INCLUDE Irvine32.inc
.data
msg_reach_end	BYTE	"You have reached the end position.", 0

maze	BYTE	"################"
	BYTE	"#..............#"
	BYTE	"#..............#"
	BYTE	"#..###.........#"
	BYTE	"#....########..#"
	BYTE	"#....#.........#"
	BYTE	"#....#..####...#"
	BYTE	"#....#.....#...#"
	BYTE	"#....#.....#####"
	BYTE	"#....#.........#"
	BYTE	"#....########..#"
	BYTE	"#.......#......#"
	BYTE	"#.......#...####"
	BYTE	"#.......#......#"
	BYTE	"#.......#......#"
	BYTE	"################"
SIZEX = SIZEOF maze
SIZEY = ($ - maze) / SIZEX
STARTX = 1
STARTY = 1
ENDX = SIZEX - 2
ENDY = SIZEY - 2

posX	DWORD	STARTX
posY	DWORD	STARTY

.code
main	PROC
	call	Clrscr

	mov	ebx, SIZEX	; draw the maze
	mov	edx, SIZEY
	mov	esi, OFFSET maze
	call	draw_maze	; you have to implemented this function


	mov	ebx, STARTX	; draw the player
	mov	edx, STARTY
	call	draw_player	; you have to implemented this function

	mov	posX, STARTX	; start to play
	mov	posY, STARTY
	call	play		; you have to implemented this function

	mov	DH, 24		; show final message
	mov	DL, 0
	call	GotoXy

	mov	eax, white
	call	SetTextColor
	call	WaitMsg

	exit
main	ENDP


;;;;;;;;;;;;;;;;;;;;;;;
draw_maze PROC

mov al, BYTE PTR [esi]
mov ecx, SIZEY


L2:
	push ecx
	mov ecx, SIZEX
L1:
	
	
	call	WriteChar
	add esi, 1
	mov al, BYTE PTR [esi]
	loop L1

	pop ecx
	call Crlf

	loop L2

	ret
draw_maze ENDP


;;;;;;;;;;;;;;;;;;;;;;;;
draw_player PROC

mov ax, 4
mov dh, STARTX
mov dl, STARTY


call Gotoxy
call SetTextColor
call WriteChar
call Gotoxy

ret
draw_player ENDP

;;;;;;;;;;;;;;;;;;;;;;;
play PROC


conti:


call Readchar

;;;;;;;;;;;;;;;;;;
	;;;;;;;;;;;;;;;;;
	mov	esi, OFFSET maze
	cmp ax, 4800h
	je UG
	cmp al, 'W'
	je UG
	cmp al, 'w'
	je UG
	cmp ax, 5000h
	je DG
	cmp al, 'S'
	je DG
	cmp al,'s'
	je DG
	cmp ax, 4B00h
	je LG
	cmp al, 'A'
	je LG
	cmp al, 'a'
	je LG
	cmp ax, 4D00h
	je RG
	cmp al, 'D'
	je RG
	cmp al, 'd'
	je RG

	UG:
	push eax
	cmp posY, 0
	je conti
	mov edx, SIZEX
	mov eax, posY
	dec eax
	push ecx
	mov ecx, eax
	mov ebx, ecx
	mov eax, edx
	mul ecx
	add eax, posX
	add esi, eax
	mov al, BYTE PTR [esi]
	pop ecx
	cmp al, '#'
	pop eax
	je conti
	jne AFT

	DG:
	push eax
	cmp posY, SIZEY
	je conti
	mov edx, SIZEX
	mov eax, posY
	inc eax
	push ecx
	mov ecx, eax
	mov ebx, ecx
	mov eax, edx
	mul ecx
	add eax, posX
	add esi, eax
	mov al, BYTE PTR [esi]
	pop ecx
	cmp al, '#'
	pop eax
	je conti
	jne AFT

	LG:
	push eax
	cmp posX, 0
	je conti
	mov edx, SIZEX
	mov eax, posY
	push ecx
	mov ecx, eax
	mov ebx, ecx
	mov eax, edx
	mul ecx
	add eax, posX
	dec eax
	add esi, eax
	mov al, BYTE PTR [esi]
	pop ecx
	cmp al, '#'
	pop eax
	je conti
	jne AFT

	RG:
	push eax
	cmp posX, SIZEX
	je conti
	mov edx, SIZEX
	mov eax, posY
	push ecx
	mov ecx, eax
	mov ebx, ecx
	mov eax, edx
	mul ecx
	add eax, posX
	inc eax
	add esi, eax
	mov al, BYTE PTR [esi]
	pop ecx
	cmp al, '#'
	pop eax
	je conti
	jne AFT
	;;;;;;;;;;;;;;;;;
AFT:
mov dh, BYTE PTR posY
mov dl, BYTE PTR posX
push ax
mov ax, white
call SetTextColor
mov al, '.'
call WriteChar
call Gotoxy
pop ax
;;;;;;;;;;;;;;;;



;;;;;;;
cmp ax, 4800h
je U
cmp al, 'W'
je U
cmp al, 'w'
jne NU
U:
dec posY
dec dh
;;;;;;
NU:
cmp ax, 5000h
je D
cmp al, 'S'
je D
cmp al,'s'
jne ND
D:
inc posY
inc dh
;;;;;;
ND:
cmp ax, 4B00h
je L
cmp al, 'A'
je L
cmp al, 'a'
jne NL
L:
dec posX
dec dl
;;;;;;
NL:
cmp ax, 4D00h
je R
cmp al, 'D'
je R
cmp al, 'd'
jne NR
R:
inc posX
inc dl
;;;;;
NR:
cmp ax, 011Bh
je FIN



;;;;;;;;;;;
mov ax, red
call SetTextColor
call Gotoxy
mov ax, 4
call WriteChar
call Gotoxy
;;;;;;;;;;;;

cmp posX, ENDX
jne conti
cmp posY, ENDY
jne conti
mov dl, 0
mov dh, SIZEY
call Gotoxy
mov edx, OFFSET msg_reach_end
call WriteString

FIN:
ret
play ENDP
END main