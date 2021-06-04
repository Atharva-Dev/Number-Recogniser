import tkinter
from tkinter import *
from PIL import Image, ImageDraw
import os
import os.path
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size = 5)
        self.conv2 = nn.Conv2d(10,30,kernel_size = 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(480, 200)
        self.fc2 = nn.Linear(200,300)
        self.fc3 = nn.Linear(300, 50)
        self.out = nn.Linear(50, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        x = F.dropout(x, training=True)
        x = x.view(-1, 480)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.out(x)

        x = F.log_softmax(x)

        return x

class UI:
    x =None
    y =None
    isClicked = False

    def __init__(self):
        self.loadBrain()
        self.root = Tk()
        wx=600
        wy=380
        self.root.geometry('{}x{}'.format(wx,wy))
        self.root.title("Number Recognizer")
        self.root.configure(bg='grey')
        self.img = Image.new('L',(280,280),0)
        self.im_draw = ImageDraw.Draw(self.img)
        self.createCanvas()
        self.createButtons()
        self.output = Label(self.root, fg='black', text='<----\nDraw a number\nbetween 0-9', bg='grey', font=(None, 15))
        self.output.grid(row=3, column=2, rowspan=3, columnspan=2, padx=10)
        self.root.mainloop()
        if os.path.isfile('.in_image.JPEG'): os.remove('.in_image.JPEG')

    def loadBrain(self):
        self.brain = NeuralNet()
        self.brain.load_state_dict(torch.load('./brain.pth'))
        self.brain.eval()

    def createCanvas(self):
        self.canvas = Canvas(self.root, bg='black', width=280, height=280)
        self.canvas.bind("<Motion>", self.draw)
        self.canvas.bind("<ButtonPress-1>", self.pressed)
        self.canvas.bind("<ButtonRelease-1>", self.released)
        self.canvas.grid(row=1, column=1, rowspan=5, padx = 30,pady = 30)

    def draw(self,event=None):
        if self.isClicked :
            if self.x is not None and self.y is not None:
                w=10

                event.widget.create_oval(self.x-w,self.y-w, self.x+w, self.y+w, fill='white', outline = 'white')
                self.im_draw.ellipse([self.x-w,self.y-w, self.x+w, self.y+w], fill='white', outline = 'white')
                self.im_draw.ellipse([self.x-1.4*w,self.y-1.4*w, self.x+1.4*w, self.y+1.4*w], fill='#999999', outline = '#999999')


            self.x = event.x
            self.y = event.y

    def pressed(self, event=None):
        self.isClicked = True

    def released(self, event=None):
        self.isClicked = False
        self.x = None
        self.y = None

    def createButtons(self):
        self.clear = Button(self.root, command= self.clearCanvas, text='Clear Canvas')
        self.go = Button(self.root, command = self.recognize, text='Recognize \nthe number')

        self.clear.grid(row=1, column=2, padx=20, pady=10)
        self.go.grid(row=2, column=2, padx=20)

    def clearCanvas(self):
        self.canvas.delete('all')
        self.output['text']='<----\nDraw a number\nbetween 0-9'
        self.canvas.bind("<Motion>", self.draw)
        self.canvas.bind("<ButtonPress-1>", self.pressed)
        self.canvas.bind("<ButtonRelease-1>", self.released)
        self.img = Image.new('L',(280,280),0)
        self.im_draw = ImageDraw.Draw(self.img)
        if os.path.isfile('.in_image.JPEG'): os.remove('.in_image.JPEG')

    def recognize(self):
        self.canvas.unbind("<Motion>")
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.img.resize((28,28)).save('.in_image.JPEG')
        image = Image.open('.in_image.JPEG')
        image_to_be_recognised = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])(image)

        image_to_be_recognised = image_to_be_recognised.unsqueeze(0)
        self.brain.eval()
        out = self.brain(image_to_be_recognised)
        self.displayResult(out)

    def displayResult(self, out):

        percentage = (out.max().exp().item())
        result = out.argmax().item()

        self.output['text'] = 'I am {:.2f}% \nsure that the \nnumber is {} '.format(percentage*100, result)







if __name__ == "__main__":
    UI()
