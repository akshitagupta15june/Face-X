# Face Recognition Using Kendryte K210 dock

Using the cheap K210 doc , we can detect objects and use it for Facial recognition attendance systems. One can program it using microPython or with arduino code. The dock comes with  a 24-pin interface lcd screen(320x240), an ov2640 camera. It has a type c port , and supports i2C , SPI , UART , APN protocols. 
Now , for this project , we have a Kmodel. We need to flash it on the board , but before that we need to flash the bin file on the board. To flash these files , we need to install [Kflash GUI (click to download)](https://github.com/sipeed/kflash_gui/releases).
After this we need to install the [MaixPy ide (click to download)](http://dl.sipeed.com/MAIX/MaixPy/ide/). After downloading , load the python file , and then connect the board to the computer , and connect through serial. Then run the script and it should work. We can load the script to the board , and we can use it anywhere with a power supply , no need to connect it to the computer. 

***Prerequisites*** -


The board - 

* [Sipeed M1 Dock](https://roborium.com/sipeed-m1-dock-suit-24-lcd-ov2640?search=sipeed)
* [Sipeed M1w Dock with wifi](https://roborium.com/sipeed-m1w-dock-suit-24-lcd-ov2640?search=sipeed)

Driver - CH340


<img src="https://roborium.com/image/cache/catalog/Products/2019/SDSMDSKDBRAB%201-771x1000.jpg" height=400 width=400 align="center" >

For more information , visit the [Maixpy Sipeed website.](https://maixpy.sipeed.com/en/)
