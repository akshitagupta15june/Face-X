#include <Servo.h>
#include <LiquidCrystal.h>

#define LED_PIN_GRE 8
#define LED_PIN_RED 12
#define LED_PIN_BUZZ 6

Servo x, y;
int width = 640, height = 480;  // total resolution of the video
int xpos = 120, ypos = 120;     // initial positions of both Servos

const int rs = 7, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

void setup() {
  lcd.begin(16, 2);
  lcd.print("Mask");

  pinMode(LED_PIN_GRE, OUTPUT);
  digitalWrite(LED_PIN_GRE, LOW);

  pinMode(LED_PIN_RED, OUTPUT);
  digitalWrite(LED_PIN_RED, LOW);

  pinMode(LED_PIN_BUZZ, OUTPUT);

  Serial.begin(9600);
  x.attach(9);
  y.attach(10);
  // Serial.print(width);
  //Serial.print("\t");
  //Serial.println(height);
  x.write(xpos);
  y.write(ypos);


}
const int angle = 5;  // degree of increment or decrement

void loop() {



  tone(LED_PIN_BUZZ, 4000);  // Send 1KHz sound signal...
                             //  delay(1000);        // ...for 1 sec
  noTone(LED_PIN_BUZZ);      // Stop sound...
                             //  delay(1000);


  if (Serial.available() > 0) {
    int x_mid = 0;
    int y_mid = 0;
    if (Serial.read() == 'X') {
      x_mid = Serial.parseInt();  // read center x-coordinate
      Serial.print(x_mid + " " + y_mid);
      if (Serial.read() == 'Y')
        y_mid = Serial.parseInt();  // read center y-coordinate
    }

    if (Serial.read() == 'P') {
      int result = Serial.parseInt();
      if (result == 0) {
        digitalWrite(LED_PIN_GRE, LOW);
        digitalWrite(LED_PIN_RED, HIGH);
      } else {
        digitalWrite(LED_PIN_GRE, HIGH);
        digitalWrite(LED_PIN_RED, LOW);
      }
    }
    if (x_mid > width / 2 + 30)
      xpos += angle;

      delay(1000);

    if (x_mid < width / 2 - 30)
      xpos -= angle;
    if (y_mid > height / 2 - 30)
    {
      ypos += angle;
      delay(1000);
    }
    if (y_mid < height / 2 + 30)
      ypos -= angle;
    
    if (xpos >= 180)
      xpos = 0;
    else if (xpos <= 0)
      xpos = 0;
    if (ypos >= 180)
      ypos = 180;
    else if (ypos <= 0)
      ypos = 0;

    x.write(xpos);
    y.write(ypos);

    // used for testing
    Serial.print("\t");
    Serial.print(x_mid);
    Serial.print("\t");
    Serial.println(y_mid);
  }
}
