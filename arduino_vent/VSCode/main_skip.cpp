#include <Arduino.h>

#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27,20,4);  // set the LCD address to 0x27 for a 16 chars and 2 line display

//BUTTONS/LEDS/SWITCHES
#define START_BTN_PIN      2
#define LIMIT_SWITCH_PIN   3
#define CONFIG_BTN_PIN     10
#define OK_BTN_PIN         12
#define START_LED_PIN      7

//STEPPER CONFIG
#define STEPPER_ENABLE     4
#define STEPPER_DIR        5
#define STEPPER_STEP       6
#define STEPPER_INVERT_DIR LOW    //Change this to Invert the stepper direction

//POT CONFIG
#define VOLUME_POT              A0  // Amount to compress the AmbuBag 200mL -1L
#define BREATHS_PER_MIN_POT     A1  // Duty cycle 6-30 per minute
#define IE_RATION_POT           A2  // Ratio of Inspiratory to Expiratory time
// #define INSPIRATORY_TIME_POT    A3  // Tweak Inspiratory time, ie time to compress bag


// ISR TO ALERT WHEN LIMIT IS HIT
boolean limitActived = false; 
void limitTriggered_ISR()
{
  limitActived = true;
}

// ISR TO ENABLE/DISABLE MACHINE
boolean startEnabled = false;
void startTriggered_ISR()
{
  startEnabled = !startEnabled; //THIS MIGHT NEED TO BE DEBOUNCED <= AND LONGPRESS ADDED TO LOCK SETTINGS
  digitalWrite(START_LED_PIN, startEnabled);
}

int steps = 0;              // Current Postion of Stepper Motor
int stepsUpperLimit = 3000; // Upper Limit of Stepper Motor <= should be configured and stored in EEPROM
byte setupState = 0;        // State to store calibration and setup
boolean lockEnabled = false;

void lcdInit()
{
  lcd.init();
  lcd.backlight();
  lcd.home ();                      // Go to the home location
  lcd.setCursor (0, 0);
  lcd.print(F("* Ventilator v0.90 *"));  
  lcd.setCursor (0, 2);
  lcd.print(F("Hold button to setup"));
}

//Double tap an AnalogRead to get a cleaner ADC read
int cleanRead(byte pin)
{
  analogRead(pin);
  delay(2);
  return analogRead(pin);
}

#define VOLUME_MIN 200
#define VOLUME_MAX 1000 //This should be entered in Calibration!!!!!****************************************
#define VOLUME_INCREMENTS 25
//Reads the Volume Pot, maps the reading to the above limits and increments by set amount
uint16_t getVolume()
{
  int volumeIncrements = (VOLUME_MAX - VOLUME_MIN) / VOLUME_INCREMENTS;
  byte volumeReading = map(cleanRead(VOLUME_POT), 0, 1023, 0, volumeIncrements);
  return (volumeReading * VOLUME_INCREMENTS) + VOLUME_MIN;
}


#define BREATHS_PER_MIN_MIN 6
#define BREATHS_PER_MIN_MAX 40
//Reads the BPM Pot, maps the reading to the above limits, contrains to within machines limits
byte getBreathsPerMiute(byte restrictedMax)
{
  byte bpm = map(cleanRead(BREATHS_PER_MIN_POT), 0, 1023, BREATHS_PER_MIN_MIN, BREATHS_PER_MIN_MAX);
  return constrain(bpm, BREATHS_PER_MIN_MIN, restrictedMax);
}

//Increments by 25 ie 1:(IE_RATIO/100)  1:1, 1:1.25 etc
#define IE_RATIO_MIN 100
#define IE_RATIO_MAX 500
//Reads the I:E Pot, maps the reading to the above limits, contrains to within calculated limits
uint16_t getIERatio(int restrictedMax)
{
  uint16_t ie =  map(cleanRead(IE_RATION_POT), 0, 1023, IE_RATIO_MIN, IE_RATIO_MAX);
  return constrain(ie, IE_RATIO_MIN, restrictedMax);
}

//Step the stepper 1 step
void slowStep(int delayTime)
{
  int halfDelay = delayTime / 2;
  digitalWrite(STEPPER_STEP,HIGH); // Output high
  delayMicroseconds(halfDelay);    // Wait
  digitalWrite(STEPPER_STEP,LOW); // Output low
  delayMicroseconds(halfDelay);   // Wait
}

/**********************************************************************
* Handle any Button Presses - with Debounce
**********************************************************************/
#define DEBOUNCE_DELAY 50
void handleBTN()
{
  static bool lastState = digitalRead(OK_BTN_PIN);
  static bool actionedState = lastState;

  bool currentState = digitalRead(OK_BTN_PIN);

 
  //Check if Button Changes state
  static unsigned long btnToggleTime = millis();
  if (currentState != lastState) {
    btnToggleTime = millis();
  }
  lastState = currentState;

  //Check if the state has stayed the same way
  if (millis() - btnToggleTime > DEBOUNCE_DELAY) {
    // whatever the reading is at, it's been there for longer than the debounce
    // delay, so take it as the actual current state:
    if(currentState != actionedState){
      if(currentState == HIGH){
        //Do something when the button becomes HIGH, or Released
      } else {
        //Do something when the button becomes LOW, or Pressed
        setupState++;
        
        Serial.println("BTN");
      }
      actionedState = currentState;
    } 
  }
  
}

#define REFRESH_RATE 400
//Display Calibration date on LCD at above refresh rate
void displayPos (int value, boolean displaySteps = false) 
{
  static unsigned long lastRefreshTime = 0;
  unsigned long timeNow = millis();
  if (timeNow - lastRefreshTime > REFRESH_RATE) {
    lcd.setCursor( 0, 3 );
    lcd.print(F("                    "));
    lcd.setCursor( 0, 3 );
    lcd.print(value);
    if(displaySteps) {
      lcd.setCursor( 15, 3 );
      lcd.print(steps);
    }
    lastRefreshTime = timeNow;
  }
}

void clearLCD()
{
  lcd.setCursor (0, 0);
  lcd.print(F("                    "));
  lcd.print(F("                    "));
  lcd.print(F("                    "));
  lcd.print(F("                    "));
  lcd.setCursor (0, 0);
}


void handleCalibrate()
{
  //Variables only used in Calibration
  int amountToBreath = 2000;
  int speedToStepAt = 2000;
  static unsigned long lungInflationTime = 0;
  static byte lastState = 0;
  
  while (setupState < 10) {
    handleBTN();              //Moves between state!
    int potReading = cleanRead(VOLUME_POT);
    switch (setupState) {
      case 0:
        Serial.println(F("CALIBRATE Routine"));
        setupState++;
      case 1:
        if (lastState != setupState) {
          Serial.println(F("Set Volume knob to 0"));
    
          clearLCD();
          lcd.print(F("***** CALIBRATE ****"));
          lcd.setCursor (0, 1);
          lcd.print(F("*** Set VOL to 0 ***"));
          lcd.setCursor (0, 2);
          lcd.print(F("Push Btn to continue"));
        }
        displayPos(map(potReading, 0, 1023, -5, 5));
        break;
      case 2:    // Moves ARM down, compressing bag until it hits limit switch
        if (lastState != setupState) {
          Serial.println(F("Compressing bag fully"));
          digitalWrite(STEPPER_ENABLE, HIGH);
          lcd.setCursor (0, 0);
          lcd.print(F("                    "));
          lcd.setCursor (0, 1);
          lcd.print(F("                    "));
          lcd.setCursor (0, 0);
          lcd.print(F("Compressing bag"));
        }

        digitalWrite(STEPPER_DIR, STEPPER_INVERT_DIR);
        while(digitalRead(LIMIT_SWITCH_PIN)){
            slowStep(800);
            steps--;
        }
        steps = 0;
        break;
      case 3:    // Moves ARM up, inflating bag until it hits coded limit
        if (lastState != setupState) {
          Serial.println(F("Set Top, Allow bag to inflate fully"));
          lcd.setCursor (0, 0);
          lcd.print(F("                    "));
          lcd.setCursor (0, 0);
          lcd.print(F("Inflating bag"));
        }
        digitalWrite(STEPPER_DIR, !STEPPER_INVERT_DIR);
        while(steps <= stepsUpperLimit) {  
          slowStep(800);
          steps++;
        }
        break;   
      case 4:    // Oscillates Bag at user adjustable depth and speed to find different volumes
        if (lastState != setupState) {
          Serial.println("Map Volume - using https://youtu.be/cy4kzOeLD5E");
          clearLCD();
          lcd.print(F("VOL = Volume"));
          lcd.setCursor (0, 1);
          lcd.print(F("BPM = Speed"));
          lcd.setCursor (0, 2);
          lcd.print(F("Find  200mL"));
        }

        digitalWrite(STEPPER_DIR, STEPPER_INVERT_DIR);
        amountToBreath = map(cleanRead(VOLUME_POT), 0, 1023, 20, 3000);
        speedToStepAt = map(cleanRead(BREATHS_PER_MIN_POT), 0, 1023, 220, 3000);
        lungInflationTime = millis();
        while(steps > amountToBreath){
            if(digitalRead(LIMIT_SWITCH_PIN)){
              slowStep(speedToStepAt);
              steps--;
            }
        }
        displayPos(millis() - lungInflationTime, true);

        delay(1000);

        digitalWrite(STEPPER_DIR, !STEPPER_INVERT_DIR);
        while(steps < stepsUpperLimit){
            slowStep(speedToStepAt*2);
            steps++;
        }
        
        delay(1000);
        
        break; 
      case 5:    // Move between Top and Bottom Limits
        //TODO:  IMPLMENT FOR ALL OTHER VOLUMES REQUIRED 200ml to Max @ 100ml increments
        break; 
    }
    lastState = setupState;
  }
}


//POSSIBLE ROLLING AVERAGE THESE
int requiredVolume = 0;
int stepsForRequiredVolume = 0;
int requiredBPM = 0;
int requiredIERatio = 0;
int calculatedInspiratoryTime = 0;
int calculatedExpiratoryTime = 0;

int machineRestrictedBPM = BREATHS_PER_MIN_MAX;
int machineRestrictedIERation = IE_RATIO_MAX;


void handleSettings()
{
  //Volume and machine specs limit BPM and IE ratio
  requiredVolume = getVolume();
  //BPM further limits IE Ratio

  //  60000 / Max Time to Produce Volume * 2 (1:1 Ratio)
  //  60000 / 800mS * 2 = 60000 / 1600 = 37.5BPM

  // machineRestrictedBPM = 6000 / (mapToSteps(requiredVolume)*stepTime)*2

  //DUMMYS TO SIMULATE MAX TIME TO REACH VOLUME & AMOUNT OF STEPS REQUIRED
  uint16_t maxTimeToReachVolume = requiredVolume; //DUMMY DUMMY DUMMY DUMMY DUMMY DUMMY DUMMY DUMMY DUMMY DUMMY
  stepsForRequiredVolume = requiredVolume;  //look up steps for required Volume!!!!

  machineRestrictedBPM = 60000 / (maxTimeToReachVolume * 2);

  requiredBPM = getBreathsPerMiute(machineRestrictedBPM);

  //Calculation examples
  //   mSPerBreath = 60000/requiredBPM;
  //   1600 = 60000/37.5
  // 1600 - Max Time to Produce Volume 
  // 1600 - 800 = 800 ExpiratoryTimeRemaining
  // (ExpiratoryTimeRemaining / Max Time to Produce Volume)  * 100
  // (800.0 / 800.0) * 100 = 100 (1:1 Ratio)

  // 60000/20 = 3000
  // 3000 - 800 = 2200 ExpiratoryTimeRemaining
  // (2200.0 / 800.0) * 100 = 275 (1:2.75 Ratio)

  // 60000/40 = 1500 mSPerBreath
  // 1500 - 700 = 900 ExpiratoryTimeRemaining
  // (900 / 700) * 100 = 128 (1:1.25 Ratio)


  //mSPerBreath = Total Inspiration & Expiratory Time
  uint16_t mSPerBreath = 60000/requiredBPM;
  uint16_t expiratoryTimeRemaining = mSPerBreath-maxTimeToReachVolume;
  machineRestrictedIERation = ((float)expiratoryTimeRemaining / (float)maxTimeToReachVolume) * 100.0;

  requiredIERatio = getIERatio(machineRestrictedIERation);

  float mSPerRatio = ((float)mSPerBreath/(100.0+(float)requiredIERatio));
  calculatedInspiratoryTime = mSPerRatio * 100;
  //calculatedExpiratoryTime = mSPerRatio * requiredIERatio;  //Incurs rounding issues - DON'T USE
  calculatedExpiratoryTime = mSPerBreath - calculatedInspiratoryTime;

  // Serial.println("");
  // Serial.print(maxTimeToReachVolume);
  // Serial.print(" : ");
  // Serial.print(mSPerBreath);
  // Serial.print(" : ");
  // Serial.print(expiratoryTimeRemaining);
  // Serial.print(" : ");
  // Serial.print(machineRestrictedIERation);
  // Serial.print(" : ");
  // Serial.print(requiredIERatio);
  // Serial.print(" : ");
  // Serial.print(mSPerRatio);
  // Serial.print(" : ");
  // Serial.print(calculatedInspiratoryTime);
  // Serial.print(" : ");
  // Serial.println(calculatedExpiratoryTime);
  // delay(1000);

  //Calculation examples
  // 60000/BPM = mSPerBreath
  // mSPerBreath / (100+requiredIERatio) = calculatedRatioTime
  // calculatedRatioTime * 100 = calculatedInspiratoryTime
  // calculatedRatioTime * requiredIERatio = calculatedExpiratoryTime

  // 30 BPM @ 1:1 Ratio
  // 60000/30 BPM = 2000mS Per Breath
  // 2000/(100+100) = 10 Ratio Time
  // 10 * 100 = 1000 calculatedInspiratoryTime
  // 10 * 100 = 1000 calculatedExpiratoryTime

  // 30 BPM @ 1:1.5 Ratio
  // 60000/30 BPM = 2000mS Per Breath
  // 2000/(100+150) = 8 Ratio Time
  // 8 * 100 = 800 calculatedInspiratoryTime
  // 8 * 150 = 1200 calculatedExpiratoryTime
}

void handleScreen()
{
  static int lastVolume = 0;
  static int lastBPM = 0;
  static int lastIERation = 0;
  if(lastVolume != requiredVolume || lastBPM != requiredBPM || lastIERation != requiredIERatio){
    lcd.setCursor(0,3);
    lcd.print(F("                    "));
    lcd.setCursor(0,3);
    lcd.print(requiredVolume);
    lcd.print("mL");


    lcd.setCursor(9,3);
    lcd.print(requiredBPM);

    lcd.setCursor(14,3);
    lcd.print("1:");
    lcd.print((float)requiredIERatio/100.0);

    lcd.setCursor(11,0);
    lcd.print("         ");
    lcd.setCursor(11,0);
    lcd.print(calculatedInspiratoryTime);
    lcd.print(":");
    lcd.print(calculatedExpiratoryTime);

    lastVolume = requiredVolume;
    lastBPM = requiredBPM;
    lastIERation = requiredIERatio;
  }
}

void breath()
{
  while(steps > stepsForRequiredVolume){
    if(digitalRead(LIMIT_SWITCH_PIN)){
      slowStep(calculatedInspiratoryTime / stepsForRequiredVolume);
      steps--;
    } else {
      //ALARM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WE SHOULD NOT HIT LIMIT
    }
    if(startEnabled == false) return; //EXIT IF START IS DISABLED
  }
  digitalWrite(STEPPER_DIR, !STEPPER_INVERT_DIR);
  while(steps < stepsUpperLimit){
    slowStep(calculatedExpiratoryTime / stepsForRequiredVolume);
    steps++;
    if(startEnabled == false) return; //EXIT IF START IS DISABLED
  }
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(OK_BTN_PIN, INPUT_PULLUP);
  pinMode(CONFIG_BTN_PIN, INPUT_PULLUP);
  pinMode(STEPPER_STEP, OUTPUT);
  pinMode(STEPPER_DIR, OUTPUT);
  pinMode(STEPPER_ENABLE, OUTPUT);
  pinMode(LIMIT_SWITCH_PIN, INPUT_PULLUP);
  pinMode(START_BTN_PIN, INPUT_PULLUP);
  pinMode(START_LED_PIN, OUTPUT);
  attachInterrupt(digitalPinToInterrupt(LIMIT_SWITCH_PIN), limitTriggered_ISR, FALLING);
  attachInterrupt(digitalPinToInterrupt(START_BTN_PIN), startTriggered_ISR, FALLING);
  lcdInit();

  while(millis() < 4000){  //Time allowed to enter calibration upon start up
    handleBTN();
    if(!digitalRead(CONFIG_BTN_PIN)){
      Serial.println("Entering Setup");
      setupState = 0;
      handleCalibrate();  
    }
  }

  clearLCD();
}


void loop() {
  if(limitActived){
    Serial.println("Limit Hit");
    limitActived = false;
  }
  // if(lockEnabled == false){
    handleSettings();
    handleScreen();
  // }
  if(startEnabled == true){
    breath();
  }
}


