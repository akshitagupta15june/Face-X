<p style="text-align:center;" align="center"><a href="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-using-IOT/readme.md"><img align="center" style="margin-bottom:20px;" src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-using-IOT/Images/Untitled1.png"  width="70%" /></a><br /><br /></p>

# Face Recognition based Authentication System using IoT
<img align="right" height="460px" src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-using-IOT/Images/aws-header.png">

## 1.Introduction 
One of the most hyped technologies in recent years has been the `Internet of Things (IoT)`, a trend that has entered our consumer lives via home monitoring systems, wearable devices, connected cars, and remote health care. These advancements have been, in large part, attributed to two factors: the expansion of `networking capabilities` and the availability of lower-cost devices. In the vision market, by comparison, these same factors have been key challenges that have instead slowed the adoption of IoT.
The increasing number of imaging and data devices within an inspection system, combined with more advanced sensor capabilities, poses a growing bandwidth crunch in the vision market. Cost is also a significant barrier for the adoption of IoT, particularly when considering an evolutionary path toward incorporating machine learning in inspection systems.

### **Evolving toward IoT and AI**

IoT promises to bring new cost and process benefits to machine vision and to pave the path toward integrating `artificial intelligence (AI)` and machine learning into `inspection systems`.

The proliferation of consumer IoT devices has been significantly aided by the availability of lightweight communication protocols (Bluetooth, MQTT, and Zigbee, to name just a few) to share low-bandwidth messaging or beacon data. These protocols provide “good enough” connectivity in applications when delays are acceptable or unnoticeable. For example, you likely wouldn’t notice if your air conditioner took a few seconds to start when you got home.

In comparison, imaging relies on `low-latency`, uncompressed data to make a real-time decision. Poor data quality or delivery can translate into costly production halts or secondary inspections 

<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-using-IOT/Images/Iot1.png" height="400px" align="left"/>

`Embedded smart devices` integrate off-the-shelf sensors and processing platforms to enable compact, lower-power devices that can be more easily networked in IoT applications


Traditionally, inspection has relied on a camera or sensor transmitting data back to a central processor for analysis. For new IoT applications that integrate numerous image and data sources — including hyperspectral and 3D sensors outputting data in various formats — this approach poses a bandwidth challenge.

To help solve the impending bandwidth crunch, designers are investigating smart devices that process data and make decisions at the edge of the inspection network. These devices can take the form of a smart frame grabber that integrates directly into an existing inspection network or a compact sensor and embedded processing board that bypass a traditional camera.

<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-using-IOT/Images/MachineVision.png" height="380px" align="right"/>

Embedded smart devices enable more sophisticated processing at the sensor level. Key to this has been the introduction of lower-cost, compact embedded boards with processing power required for real-time image analysis. Embedded smart devices are ideal for repeated and automated robotic processes, such as edge detection in a pick-and-place system.

Like the smart frame grabber, embedded smart devices offer a straightforward path to integrating AI into vision applications. With local and cloud-based processing and the ability to share data between multiple smart devices, AI techniques can help train the computer model to identify objects, defects, and flaws while supporting a migration toward self-learning robotics systems.


## 2.Computer vision and the `cloud`

The cloud and having access to a wider data set will play important roles in bringing IoT to the vision market. Traditionally, production data has been limited to a facility. There is now an evolution toward cloud-based data analysis, where a wider data set from a number of global facilities can be used to improve inspection processes.

Instead of relying on rules-based programming, vision systems can be trained to make decisions using algorithms extracted from the collected data. With a scalable cloud-based approach to learning from new data sets, `AI and machine learning processing algorithms` can be continually updated and improved to drive efficiency. Smart frame grabbers and embedded imaging devices provide a straightforward entry point to integrating preliminary AI capabilities within a vision system.

Inexpensive cloud computing also means algorithms that were once computationally too expensive because of dedicated infrastructure requirements are now affordable. For applications such as `object recognition`, `detection`, and `classification`, the learning portion of the process that once required vast computing resources can now happen in the cloud versus via a dedicated, owned, and expensive infrastructure. The processing power required for imaging systems to accurately and repeatedly simulate human understanding, learn new processes, and identify and even correct flaws is now within reach for any system designer.
<br></br>
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-using-IOT/Images/machinedesign1.png" height="300px" align="left"/>
## 3.What is IOT?
**The `Internet of Things (IoT) `refers to a system of interrelated, internet-connected objects that are able to collect and transfer data over a wireless network without` human intervention`.** The Internet of things describes the network of physical objects—“things”—that are embedded with sensors, software, and other technologies for the purpose of connecting and exchanging data with other devices and systems over the Internet.
      
Because of the rapid growth in the IoT space, there are a number of competing standards, tools, projects, policies, frameworks, and organizations hoping to define how connected devices communicate in the modern era. Open source and open standards will become increasingly important to ensure that devices are able to properly interconnect, as well as for the back end of processing the enormous volumes of big data that all of these devices will generate.

## 4.How do IoT devices work?

Smartphones do play a large role in the IoT, however, because many IoT devices can be controlled through an app on a smartphone. You can use your smartphone to communicate with your smart thermostat, for example, to deliver the perfect temperature for you by the time you get home from work. Another plus? This can eliminate unneeded heating or cooling while you’re away, potentially saving you money on energy costs.

IoT devices contain sensors and mini-computer processors that act on the data collected by the sensors via machine learning. Essentially, IoT devices are mini computers, connected to the internet, and are vulnerable to malware and hacking.

Machine learning is when computers learn in a similar way to humans — by collecting data from their surroundings — and it is what makes IoT devices smart. This data can help the machine learn your preferences and adjust itself accordingly. Machine learning is a type of artificial intelligence that helps computers learn without having to be programmed by someone.

That doesn’t mean your smart speaker will discuss the key points of last night’s big game with you. But your connected refrigerator may send you an alert on your smartphone that you’re low on eggs and milk because it knows you’re near a supermarket.

## 5.What are the benefits of the IoT?

The Internet of Things is designed to make our lives more convenient. Here are a few examples:

Smart bathroom scales working in tandem with your treadmill, delivering food preparation ideas to your laptop or smartphone, so you stay healthy.
Security devices monitoring your home, turning lights on and off as you enter and exit rooms, and streaming video so you can check in while you’re away.
Smart voice assistants placing your usual takeout order on command, making it a breeze to get fresh food delivered to your door.


## 6.Iot device related Information: 

### 1.`Arduino Uno`
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-using-IOT/Images/ARDUINO.png" height="300px" align="right"/>
Arduino Uno is a microcontroller board based on `8-bit ATmega328P microcontroller`. Along with ATmega328P, it consists other components such as crystal oscillator, serial communication, voltage regulator, etc. to support the microcontroller. Arduino Uno has 14 digital input/output pins (out of which 6 can be used as PWM outputs), 6 analog input pins, a USB connection, A Power barrel jack, an ICSP header and a reset button.

- How to use Arduino Board
The 14 digital input/output pins can be used as input or output pins by using `pinMode()`, `digitalRead()` and `digitalWrite()` functions in arduino programming. Each pin operate at 5V and can provide or receive a maximum of 40mA current, and has an internal pull-up resistor of 20-50 KOhms which are disconnected by default.  Out of these 14 pins, some pins have specific functions as listed below:

- `Serial Pins 0 (Rx) and 1 (Tx)`: Rx and Tx pins are used to receive and transmit TTL serial data. They are connected with the corresponding ATmega328P USB to TTL serial chip.
- `External Interrupt Pins 2 and 3`: These pins can be configured to trigger an interrupt on a low value, a rising or falling edge, or a change in value.
- `PWM Pins 3, 5, 6, 9 and 11`: These pins provide an 8-bit PWM output by using analogWrite() function.
- `SPI Pins 10 (SS), 11 (MOSI), 12 (MISO) and 13 (SCK`): These pins are used for SPI communication.
- ` In-built LED Pin 13`: This pin is connected with an built-in LED, when pin 13 is HIGH – LED is on and when pin 13 is LOW, its off.
Along with 14 Digital pins, there are 6 analog input pins, each of which provide 10 bits of resolution, i.e. 1024 different values. They measure from 0 to 5 volts but this limit can be increased by using AREF pin with analog `Reference()` function.  

`Analog pin 4 (SDA) `and`pin 5 (SCA)` also used for TWI communication using Wire library.
Arduino Uno has a couple of other pins as explained below:

- `AREF`: Used to provide reference voltage for analog inputs with `analogReference()` function.
-` Reset Pin`: Making this pin LOW, resets the microcontroller

###  2.`Raspberry pi`
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-using-IOT/Images/MacV.png" align="right"/>

`Raspberry Pi` is the name of a series of single-board computers made by the Raspberry Pi Foundation, a UK charity that aims to educate people in computing and create easier access to computing education.

The `Raspberry Pi launched in 2012`, and there have been several iterations and variations released since then. The original Pi had a single-core 700MHz CPU and just 256MB RAM, and the latest model has a quad-core CPU clocking in at over 1.5GHz, and 4GB RAM. The price point for Raspberry Pi has always been under $100 (usually around $35 USD), most notably the Pi Zero, which costs just $5.

All over the world, people use the Raspberry Pi to learn programming skills, build hardware projects, do home automation, implement Kubernetes clusters and Edge computing, and even use them in industrial applications.

The Raspberry Pi is a very cheap computer that runs Linux, but it also provides a set of GPIO (general purpose input/output) pins, allowing you to control electronic components for physical computing and explore the `Internet of Things (IoT)`.

###   Face Recognition architecture of  Iot Device (Example Smart Attendance system)
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-using-IOT/Images/Arc.png"/>

<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-using-IOT/Images/IOT-banner2.png" height="400px" align="left"/>
<p style="clear:both;">
<h1><a name="contributing"></a><a name="community"></a> <a href="https://github.com/akshitagupta15june/Face-X">Community</a> and <a href="https://github.com/akshitagupta15june/Face-X/blob/master/CONTRIBUTING.md">Contributing</a></h1>
<p>Please do! Contributions, updates, <a href="https://github.com/akshitagupta15june/Face-X/issues"></a> and <a href=" ">pull requests</a> are welcome. This project is community-built and welcomes collaboration. Contributors are expected to adhere to the <a href="https://gssoc.girlscript.tech/">GOSSC Code of Conduct</a>.
</p>
<p>
Jump into our <a href="https://discord.com/invite/Jmc97prqjb">Discord</a>! Our projects are community-built and welcome collaboration. 👍Be sure to see the <a href="https://github.com/akshitagupta15june/Face-X/blob/master/Readme.md">Face-X Community Welcome Guide</a> for a tour of resources available to you.
</p>
<p>
<i>Not sure where to start?</i> Grab an open issue with the <a href="https://github.com/akshitagupta15june/Face-X/issues">help-wanted label</a>
</p>
**`Open Source First`**
<p>We build projects to provide learning environments, deployment and operational best practices, performance benchmarks, create documentation, share networking opportunities, and more. Our shared commitment to the open source spirit pushes Face-x projects forward.</p>
