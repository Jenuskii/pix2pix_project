from tkinter import*
from PIL import Image, ImageTk, ImageFilter
import tkinter as tk
from tkinter import filedialog as fd
from tkinter.colorchooser import*
from PIL import ImageGrab
from transform import*




im = None
tk_img = None
window = None
canvas = None
x1, y1 = None, None

TopTransformer = ImageTransformer('TOP')
ShoesTransformer = ImageTransformer('SHOES')
BagTransformer = ImageTransformer('BAG')




def paint(event):
    global x1, y1
    x1=event.x
    y1=event.y
    canvas.create_line(x1,y1,x1+1,y1+1, width=3, fill='black')
    
def clearcanvas():
    canvas.delete('all')
    
def save():
    filename = filedialog.asksaveasfilename(initialdir='/', title='Select file',
                                         filetypes=(('PPTX files', '*.pptx'), ("all files", "*.*")))
def Open():
    global im, tk_img
    fname = fd.askopenfilename()
    im = Image.open(fname)
    tk_img = ImageTk.PhotoImage(im)
    canvas.create_image(128, 128, image=tk._img)
    window.update()

def quit():
    window.quit()
    window.destroy()

def eraser(event):
    global x1, y1
    x1=event.x
    y1=event.y
    canvas.create_line(x1,y1,x1+1,y1+1, width=100, fill='white')
    
def top():
    x=canvas.winfo_rootx()
    y=canvas.winfo_rooty()
    w=canvas.winfo_width()+x
    h=canvas.winfo_height()+y
    box=(x,y,w,h)
    cap=ImageGrab.grab(box)
    cap.save('capture.png')

    img_original = Image.open('capture.png')
    img_output = TopTransformer.transform(img_original)
    TopTransformer.save_img(img_output)
    
def shoes():
    x=canvas.winfo_rootx()
    y=canvas.winfo_rooty()
    w=canvas.winfo_width()+x
    h=canvas.winfo_height()+y
    box=(x,y,w,h)
    cap=ImageGrab.grab(box)
    cap.save('capture.png')

    img_original = Image.open('capture.png')
    img_output = ShoesTransformer.transform(img_original)
    TopTransformer.save_img(img_output)
    
def bag():
    x=canvas.winfo_rootx()
    y=canvas.winfo_rooty()
    w=canvas.winfo_width()+x
    h=canvas.winfo_height()+y
    box=(x,y,w,h)
    cap=ImageGrab.grab(box)
    cap.save('capture.png')

    img_original = Image.open('capture.png')
    img_output = BagTransformer.transform(img_original)
    TopTransformer.save_img(img_output)




window = Tk()
window.geometry('333x260')
window.title('그림판')

canvas = Canvas(window, width=252, height=252, bg = 'white')
canvas.grid()

frame = Frame(window)
frame.grid()




menubar = tk.Menu(window)
filemenu = tk.Menu(menubar)

menubar.add_cascade(label='파일', menu=filemenu) 
filemenu.add_command(label='열기', command=Open) #수정
filemenu.add_command(label='저장', command=save) #수정
filemenu.add_command(label='끝내기', command=quit)




canvas.bind('<B1-Motion>', paint)
canvas.bind('<B3-Motion>', eraser)




l2=Label(window,text="모델 선택")
l2.place(x=265, y=10)

#상의
button1=Button(window,text='Top', fg='white', bg='black', command=top)
button1.place(x=277, y=33)

#신발
button2=Button(window,text='Shoes', fg='white', bg='black', command=shoes)
button2.place(x=272, y=66)

#가방
button3=Button(window,text='Bag', fg='white', bg='black', command=bag)
button3.place(x=277, y=99)

#창 초기화
button4=Button(window,text='New', fg='white', bg='black', command=clearcanvas, font=('둥근모꼴', 10))
button4.place(x=275, y=205)




window.config(menu=menubar)
window.mainloop()