#include <Arduino.h>

#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <EEPROM.h>

LiquidCrystal_I2C lcd(0x27,20,4);  // set the LCD address to 0x27 for a 16 chars and 4 line display
//lcd refresh rate
#define REFRESH_RATE 200

#define DEBUG false

//BUTTONS/LEDS/SWITCHES
#define START_BTN_PIN      2
#define LIMIT_SWITCH_PIN   3
#define CONFIG_BTN_PIN     10
#define OK_BTN_PIN         12
#define START_LED_PIN      7
#define BUZZER             8

//STEPPER CONFIGS
#define STEPPER_ENABLE     4
#define STEPPER_DIR        5
#define STEPPER_STEP       6
#define STEPPER_DIR_DOWN   LOW
#define STEPPER_DIR_UP     HIGH

//POT CONFIG
#define VOLUME_POT              A0  // Amount to compress the AmbuBag 50ml -1L
#define BREATHS_PER_MIN_POT     A1  // Duty cycle 6-30 per minute
#define IE_RATIO_POT           A2  // Ratio of Inspiratory to Expiratory time

//debounce for button
#define BTN_DEBOUNCE_DELAY 20
//debounce for pot
#define POT_MAX_VALUE 255 // stop jittering

// steps for upper most limit of arm
#define BAG_UPPER_LIMIT 3100

//BREATH VARIABLES
#define VOLUME_MIN 150
#define VOLUME_MAX 850 //This should be entered in Calibration!!!!!****************************************
#define VOLUME_INCREMENTS 50 //ml per increment
#define STEP_TO_VOLUME_INCREMENTS ((VOLUME_MAX - VOLUME_MIN) / VOLUME_INCREMENTS) + 1 //include min and max

#define BREATHS_PER_MIN_MIN 6
#define BREATHS_PER_MIN_MAX 50

// in order to get float value between 1 and 5,
// 100 and 500 are used then reading divided by 100 = ratio
// e.g. 1:1.235, 1: 2.75, etc 
#define IE_RATIO_MIN 100
#define IE_RATIO_MAX 500 

#define MIN_US_PER_STEP 200

// #define INSPIRATORY_PAUSE 0
// #define EXPPIRATORY_PAUSE 0

// Always give your config an id, useful to debug and when config layout changes
#define CONFIG_VERSION 1


//IO VARIABLES
//Postion of Stepper Motor
uint16_t steps = BAG_UPPER_LIMIT;

//temporary pot readings
uint16_t vol = 0;
uint16_t tempVolume = 0;
uint16_t tempBPM = 0;
uint16_t tempIE = 0;


//confirmed pot readings
uint16_t volumeCurrent = 0;
uint16_t BPMCurrent = 0;
uint16_t IECurrent = 0;

//calculations
uint16_t calcVolume = 0;
uint16_t calcBPM = 0;
uint16_t calcIE = 0;
uint16_t calculatedInspiratoryTime = 0;
uint16_t calculatedExpiratoryTime = 0;
uint16_t stepsPerBreath = 0;
uint16_t inspStepsPerSecond = 0;
uint16_t expStepsPerSecond = 0;
uint16_t inspMicroSecondsPerStep = 0;
uint16_t expMicroSecondsPerStep = 0;


//calibration index 
uint8_t calibIndex = 0;

// FLAGS
boolean okBtnFlag = false;
boolean confBtnFlag = false;
boolean lcdDis = false;
boolean limitActived = false;
boolean errFlag = true;
boolean calibDone = false;

//Start button flag and timer
volatile boolean startEnabled = false;
volatile unsigned long lastStartPress = 0;


//EEPROM STRUCT
struct configStruct {
  //ADD EVERYTHING HERE YOU WANT USERS TO BE ABLE TO CONFIGURE
  //ie VOLUME_MAX, BAG_UPPER_LIMIT, etc. 
  uint16_t stepsUpperLimit;
  uint16_t stepsToVolume[STEP_TO_VOLUME_INCREMENTS]; //include max and min value
  uint8_t version; // Placed last to verify you wrote/read correctly
} config = {
  //Set defaults
  BAG_UPPER_LIMIT,
  //hard coded default increments
  //TODO - add loadConfig() based on increment setting (this array is based on 50 ml increments = 15)
  {2965, 2713, 2497, 2353, 2227, 2083, 1975, 1849, 1705, 1561, 1417, 1273, 1165, 1075, 319},
  CONFIG_VERSION
};


// OPERATING MODES
enum MODE {
  //WAIT,
  //RESET_ARM,
  STANDBY,
  POT_CONFIG,
  VOL_CONFIG,
  ML_CONFIG,
  READY,
  SET,
  RUNNING,
  ERR
};

MODE mode;


//**************************************************END OF DEFINITIONS*******************************************************



/**********************************************************************
####################### ISR's ####################################
**********************************************************************/

// ISR TO ALERT WHEN LIMIT IS HIT
void limitTriggered_ISR()
{
  limitActived = true;
}

// ISR TO ENABLE/DISABLE MACHINE
void startTriggered_ISR()
{
  unsigned long timeNow = millis();
  if(timeNow - lastStartPress > 250 ){//&& digitalRead(START_BTN_PIN) == LOW) {
    startEnabled = !startEnabled;
    digitalWrite(START_LED_PIN, startEnabled);
    digitalWrite(STEPPER_ENABLE, startEnabled);
    //starting
    if (startEnabled) {
      //only start if mode == READY
      if (mode == READY){
        mode = RUNNING;
        lcdDis = true;
      }

    //stopping
    } else {
      mode = READY;
      lcdDis = true;
    }
    
  }
  lastStartPress = timeNow;
}


/**********************************************************************
####################### FUNCTIONS ####################################
**********************************************************************/

/**********************************************************************
* Initialise LCD screen
**********************************************************************/
void lcdInit()
{
  lcd.init();
  lcd.backlight();
  lcd.home ();                      // Go to the home location
}

/**********************************************************************
* Clear lcd
**********************************************************************/
void clearLCD()
{
  lcd.setCursor (0, 0);
  lcd.print(F("                    "));
  lcd.print(F("                    "));
  lcd.print(F("                    "));
  lcd.print(F("                    "));
  lcd.setCursor (0, 0);
}

/**********************************************************************
* Save configuration variables to EEPROM
**********************************************************************/
void saveConfig() 
{
  for (uint8_t t=0; t<sizeof(config); t++) {
    EEPROM.write(t, *((char*)&config + t));
  }
  Serial.println();
  Serial.println("Config saved!");
  Serial.println();
}

/**********************************************************************
* Load configuration variables from EEPROM
**********************************************************************/
void loadConfig() 
{
  // To make sure there are settings, and they are YOURS!
  // If nothing is found it will use the default settings.
  if (EEPROM.read(sizeof(config)-1) == CONFIG_VERSION) {
    for (uint8_t t=0; t<sizeof(config); t++) {
      *((char*)&config + t) = EEPROM.read(t);
    }
  } else {
    //NO VALID CONFIG FOUND SAVING 
    //add evenly spaced increments
    //TODO - NOT WORKING,  wrong number of elements???
    // for (int i = STEP_TO_VOLUME_INCREMENTS-1; i > 0; i--){
    //   //for BAG_UPPER_LIMIT = 3600, STEP_TO_VOLUME_INCREMENTS = 50:
    //   //increments = [2965, 2713, 2497, 2353, 2227, 2083, 1975, 1849, 1705, 1561, 1417, 1273, 1165, 1075, 319]
    //   config.stepsToVolume[i] = (BAG_UPPER_LIMIT / STEP_TO_VOLUME_INCREMENTS) * i;
    //   //Serial.print((BAG_UPPER_LIMIT / STEP_TO_VOLUME_INCREMENTS) * i);
    //   //Serial.print(" ,");
    // }
    //save config
    saveConfig();
  }

  //print values of config after loading
  Serial.println();
  Serial.print("Config step to volume array: ");
  for (int i = 0; i < STEP_TO_VOLUME_INCREMENTS; i++){
    //for BAG_UPPER_LIMIT = 3600, STEP_TO_VOLUME_INCREMENTS = 50:
    //increments = [3612 ,3354 ,3096 ,2838 ,2580 ,2322 ,2064 ,1806 ,1548 ,1290 ,1032 ,774 ,516 ,258]
    Serial.print(config.stepsToVolume[i]);
    Serial.print(", ");
  }
  Serial.println();
  Serial.print("Config loaded version: ");
  Serial.println(config.version);
  Serial.print("Config upper limit: ");
  Serial.println(config.stepsUpperLimit);
  Serial.println();
}


/**********************************************************************
* Clean read pin
**********************************************************************/
int cleanRead(byte pin)
{
  return map(analogRead(pin), 0, 1023, 0, POT_MAX_VALUE);
}

/**********************************************************************
* Buzzer
**********************************************************************/
void buzzer(int ms)
{
  //Serial.println("Buzzing..");
  tone(BUZZER, 1000); // Send 1KHz sound signal...
  delay(ms);        // ...for ms duration
  noTone(BUZZER);     // Stop sound..
}


/**********************************************************************
* Slow step of stepper
**********************************************************************/
void slowStep(int delayTime)
{
  //Step the stepper 1 step
  digitalWrite(STEPPER_STEP, HIGH); // Output high
  delayMicroseconds(delayTime);    // Wait
  digitalWrite(STEPPER_STEP, LOW); // Output low
  delayMicroseconds(delayTime);   // Wait
}

/**********************************************************************
* Handle any Button Presses - with Debounce
**********************************************************************/
void handleBTN()
{
  // Debounce config button return false
  if (digitalRead(CONFIG_BTN_PIN) == LOW) { //<<<< High
    delay(BTN_DEBOUNCE_DELAY);
    if (digitalRead(CONFIG_BTN_PIN) == LOW) {
      confBtnFlag = true;
    }
  }

  // Debounce ok button return true
  if (digitalRead(OK_BTN_PIN) == LOW) {
    delay(BTN_DEBOUNCE_DELAY);
    if (digitalRead(OK_BTN_PIN) == LOW) {
      okBtnFlag = true;
    }
  }
}


/**********************************************************************
* Zero arm position
**********************************************************************/
void zeroArm() {
  // Zero arm on power on
  clearLCD();
  lcd.print("Zeroing arm position");
  lcd.setCursor(0,2);
  lcd.print("**Keep hands clear**");
  // Move arm down to switch
  digitalWrite(STEPPER_ENABLE, HIGH);
  digitalWrite(STEPPER_DIR, STEPPER_DIR_DOWN);
  while(!limitActived) {
    slowStep(200);
    //no need to decrement, possible unknown position
    //steps--;
  }
  limitActived = false;
  steps = 0;
  // Move arm to upper limit
  digitalWrite(STEPPER_DIR, STEPPER_DIR_UP);
  while(steps < config.stepsUpperLimit) {
    slowStep(200);
    steps++;
  }
  Serial.print("MAX STEPS: ");
  Serial.println(config.stepsUpperLimit);
  digitalWrite(STEPPER_ENABLE, LOW);

  //small buzz
  buzzer(100);
}


/**********************************************************************
* Set arm to last known VALID calibration position
**********************************************************************/
void resetToLast(uint16_t lastGoodVolume)
{
  //first zero arm
  zeroArm();

  // Move arm down to switch
  digitalWrite(STEPPER_ENABLE, HIGH);
  digitalWrite(STEPPER_DIR, STEPPER_DIR_DOWN);
  while(steps > lastGoodVolume) {
    slowStep(200);
    //no need to decrement, possible unknown position
    steps--;
  }
  //digitalWrite(STEPPER_DIR, STEPPER_DIR_UP);
  digitalWrite(STEPPER_ENABLE, LOW);
}


/**********************************************************************
* Update VOLUME, BPM, IE, MILLISECONS screen
**********************************************************************/
void handleScreen()
{
  //if first display, update all characters
  if(lcdDis){
    //clear first
    clearLCD();

    lcd.setCursor(1,2);
    lcd.print("VT");
    lcd.setCursor(0,3);
    lcd.print(F("                    "));
    lcd.setCursor(0,3);
    lcd.print(volumeCurrent);
    lcd.print("ml");

    lcd.setCursor(9,2);
    lcd.print("BPM");
    lcd.setCursor(9,3);
    lcd.print(BPMCurrent);

    lcd.setCursor(16,2);
    lcd.print("I/E");
    lcd.setCursor(14,3);
    lcd.print("1:");
    lcd.print((float)IECurrent/100);

    lcd.setCursor(0,0);
    lcd.print("In:Ex (ms)");
    lcd.setCursor(11,0);
    lcd.print("         ");
    lcd.setCursor(11,0);
    lcd.print(calculatedInspiratoryTime);
    lcd.print(":");
    lcd.print(calculatedExpiratoryTime);

    lcdDis = false;

  } else { 
    //otherwise only update the numbers
    //Volume
    lcd.setCursor(0,3);
    lcd.print(volumeCurrent);
    
    //BPM 
    lcd.setCursor(9,3);
    lcd.print("  ");   // clear first
    lcd.setCursor(9,3);
    lcd.print(BPMCurrent);
    
    //IE Ratio
    lcd.setCursor(14,3);
    lcd.print("1:");
    lcd.print((float)IECurrent/100);

    //IE time
    lcd.setCursor(11,0);
    lcd.print("         "); // clear first
    lcd.setCursor(11,0);
    lcd.print(calculatedInspiratoryTime);
    lcd.print(":");
    lcd.print(calculatedExpiratoryTime);
  }
}




// *********************************************************************************************************************************






void handleSettings()
{
  //read the 3 pot values
  vol = map(cleanRead(VOLUME_POT), 0, POT_MAX_VALUE, 0, STEP_TO_VOLUME_INCREMENTS);
  tempVolume = constrain(((vol * VOLUME_INCREMENTS) + VOLUME_MIN), VOLUME_MIN, VOLUME_MAX);  
  tempBPM = map(cleanRead(BREATHS_PER_MIN_POT), 0, POT_MAX_VALUE, BREATHS_PER_MIN_MIN, BREATHS_PER_MIN_MAX);
  tempIE = map(cleanRead(IE_RATIO_POT), 0, POT_MAX_VALUE, IE_RATIO_MIN, IE_RATIO_MAX);

  Serial.print("Potentiometer values: ");
  Serial.print(tempVolume);
  Serial.print(", ");
  Serial.print(tempBPM);
  Serial.print(", ");
  Serial.println(tempIE);

  //Recalculate if pots have changed
  if (tempVolume != volumeCurrent || tempBPM != BPMCurrent || tempIE != IECurrent){
    //CALCULATE - ins:exp percent
    float inspPercent = 1.00 / (1.00 + tempIE/100.00);
    float expPercent = (tempIE/100.00) / (1.00 + tempIE/100.00);

    // Serial.print("I:E = ");
    // Serial.print(ins_percent);
    // Serial.print(":");
    // Serial.println(exp_percent);

    //CALCULATE - ms per breath
    //(60sec / BPM) * 1000ms
    float msPerBreath = (60.0 / tempBPM) * 1000;
    Serial.print("ms per breath: ");
    Serial.println(msPerBreath);

    //CALCULATE - inspiratory/expiratory time in ms
    uint16_t tempCalculatedInspiratoryTime = msPerBreath * inspPercent;
    uint16_t tempCalculatedExpiratoryTime = msPerBreath * expPercent; // could be mis-calculated due to rounding?
    // Serial.print("I:E = ");
    // Serial.print(calculatedInspiratoryTime);
    // Serial.print(":");
    // Serial.println(calculatedExpiratoryTime);

    //CALCULATE - steps per breath
    uint8_t spbIndex = STEP_TO_VOLUME_INCREMENTS - ((tempVolume-VOLUME_MIN)/VOLUME_INCREMENTS);
    uint16_t tempStepsPerBreath = config.stepsToVolume[spbIndex-1];

    // for (int i = 0; i < STEP_TO_VOLUME_INCREMENTS; i++){
    //   //show the value for each
    //   Serial.print(config.stepsToVolume[i]);
    //   Serial.print(" ,");
    // }
    // Serial.print("Index: ");
    // Serial.println(spbIndex-1);
    Serial.print("Steps per breath: ");
    Serial.println(tempStepsPerBreath);

    //CALCULATE - inspiratory/expiratory steps per second
    uint16_t tempInspStepsPerSecond = (tempStepsPerBreath * 1.0 / tempCalculatedInspiratoryTime * 1.0) * 1000;
    uint16_t tempExpStepsPerSecond = (tempStepsPerBreath * 1.0 / tempCalculatedExpiratoryTime * 1.0) * 1000;

    // Serial.print("Insp steps / sec: ");
    // Serial.println(inspStepsPerSecond);
    // Serial.print("Exp steps / sec: ");
    // Serial.println(expStepsPerSecond);

    //inspiratory ms per step
    uint16_t tempInspMicroSecondsPerStep = (msPerBreath*inspPercent)/tempStepsPerBreath * 1000;
    uint16_t tempExpMicroSecondsPerStep = (msPerBreath*expPercent)/tempStepsPerBreath * 1000;

    Serial.print("Insp us / step: ");
    Serial.println(tempInspMicroSecondsPerStep);
    Serial.print("Exp us / step: ");
    Serial.println(tempExpMicroSecondsPerStep);


    //*** ONLY UPDATE VARIABLES IF MICROSECOND PER STEP > MINIMUM POSSIBLE (predefined value) *** 
    
    if(tempInspMicroSecondsPerStep >= MIN_US_PER_STEP && tempExpMicroSecondsPerStep >= MIN_US_PER_STEP){
      calculatedInspiratoryTime = tempCalculatedInspiratoryTime;
      calculatedExpiratoryTime = tempCalculatedExpiratoryTime;
      stepsPerBreath = tempStepsPerBreath;
      inspStepsPerSecond = tempInspStepsPerSecond;
      expStepsPerSecond = tempExpStepsPerSecond;
      inspMicroSecondsPerStep = tempInspMicroSecondsPerStep;
      expMicroSecondsPerStep = tempExpMicroSecondsPerStep;

      //update volume, BPM, IE to pot values
      volumeCurrent = tempVolume;
      BPMCurrent = tempBPM;
      IECurrent = tempIE;
    }
  }
}


/**********************************************************************
* BREATH
**********************************************************************/
void breath()
{
  //cycle timer
  unsigned long time = millis();
  
  // Move arm down 
  digitalWrite(STEPPER_DIR, STEPPER_DIR_DOWN);
  while(steps > (config.stepsUpperLimit - stepsPerBreath + 50)){
    slowStep(inspMicroSecondsPerStep/2);
    steps--;
    //in case of motor slip,
    if (limitActived){
      limitActived = false;
      //reset steps
      steps = 0;
      break;
    }
  }
  delay(10); //Inspiratory pause

  //check pots and update if needed
  //handleSettings();
  //handleScreen();

  // Move arm to upper limit
  digitalWrite(STEPPER_DIR, STEPPER_DIR_UP);
  while(steps < config.stepsUpperLimit) {
    slowStep(expMicroSecondsPerStep/2);
    steps++;
  }
  delay(10); //Expiratory pause

  //check we are not loosing steps (slipping)
  Serial.print("STEPS AT ORIGIN: ");
  Serial.println(steps);
  
  //print cycle time
  Serial.print("Time: ");
  Serial.println(millis() - time);

}



/**********************************************************************
* BREATH
**********************************************************************/
void setValues()
{
  uint16_t checkVOL = 0;
  uint16_t checkBPM = 0;
  uint16_t checkIE = 0;

  //print message
  //clear first
  clearLCD();
  lcd.print(F("****** Setting *****"));
  lcd.setCursor(1,2);
  lcd.print("VT");
  lcd.setCursor(0,3);
  lcd.print(F("                    "));
  lcd.setCursor(0,3);
  lcd.print(volumeCurrent);
  lcd.print("ml");

  lcd.setCursor(9,2);
  lcd.print("BPM");
  lcd.setCursor(9,3);
  lcd.print(BPMCurrent);

  lcd.setCursor(16,2);
  lcd.print("I/E");
  lcd.setCursor(14,3);
  lcd.print("1:");
  lcd.print((float)IECurrent/100);

  while(!okBtnFlag){
      
    //read the 3 pot values
    vol = map(cleanRead(VOLUME_POT), 0, POT_MAX_VALUE, 0, STEP_TO_VOLUME_INCREMENTS);
    tempVolume = constrain(((vol * VOLUME_INCREMENTS) + VOLUME_MIN), VOLUME_MIN, VOLUME_MAX);  
    tempBPM = map(cleanRead(BREATHS_PER_MIN_POT), 0, POT_MAX_VALUE, BREATHS_PER_MIN_MIN, BREATHS_PER_MIN_MAX);
    tempIE = map(cleanRead(IE_RATIO_POT), 0, POT_MAX_VALUE, IE_RATIO_MIN, IE_RATIO_MAX);

    
    //update values if changed
    if(checkVOL != tempVolume || checkBPM != tempBPM || checkIE != tempIE){
      //Volume
      lcd.setCursor(0,3);
      lcd.print(tempVolume);
      
      //BPM 
      lcd.setCursor(9,3);
      lcd.print("  ");   // clear first
      lcd.setCursor(9,3);
      lcd.print(tempBPM);
      
      //IE Ratio
      lcd.setCursor(14,3);
      lcd.print("1:");
      lcd.print((float)tempIE/100);

      //update new variables
      checkVOL = tempVolume;
      checkBPM = tempBPM;
      checkIE = tempIE;
    }

    //check btn press
    handleBTN();

  }
  //reset flag
  okBtnFlag = false;

  //update settings
  handleSettings();
      
}




/**********************************************************************
* CONFIGURATION ROUTINES
**********************************************************************/


//####################################################################
/*
VOLUME CALIBRATION CONFIGURATION
*/
void volConfig() {
  //user message to put in new bag
  if(lcdDis){
    clearLCD();
    lcd.print(F("* Ventilator v0.90 *"));
    lcd.setCursor(0,2);
    lcd.print(F("Put in new bag"));
    lcd.setCursor(0,3);
    lcd.print(F("Press OK "));

    while(!okBtnFlag){
      //spin
      handleBTN();
    }
  }
  lcdDis = false;
  okBtnFlag = false;

  // Compress bag to get 0 point
  Serial.println(F("Compressing bag fully"));
  clearLCD();
  lcd.print(F("**Compressing bag**"));
  lcd.setCursor (0, 3);
  digitalWrite(STEPPER_ENABLE, HIGH);
  digitalWrite(STEPPER_DIR, STEPPER_DIR_DOWN);
  while(!limitActived){
      slowStep(400);
      steps--;
  }
  limitActived = false;
  lcd.setCursor(0,3);
  lcd.print("Push OK to continue");
  while (!okBtnFlag) {
    handleBTN();
  }
  okBtnFlag = false;
  // Zero point found, bag fully compressed
  steps = 0;

  // Inflate Bag to set max, count steps
  // Write screen
  clearLCD();
  lcd.print(F("Set Maximum Volume "));
  lcd.setCursor(0,1);
  lcd.print(F("***Inflating Bag***"));
  lcd.setCursor(0,2);
  lcd.print(F("Press OK once bag is"));
  lcd.setCursor(0,3);
  lcd.print(F("fully inflated."));
  digitalWrite(STEPPER_DIR, STEPPER_DIR_UP);
  // Only inflate to upper limit as hardcoded
  Serial.println(okBtnFlag);
  while (steps < BAG_UPPER_LIMIT && !okBtnFlag) { //must be BAG_UPPER_LIMIT, not stepsUpperLimit
    slowStep(800);
    steps++;
    handleBTN();
  }
  // OK button pushed
  okBtnFlag = false;
  digitalWrite(STEPPER_ENABLE, LOW);

  //set the NEW UPPER LIMIT
  config.stepsUpperLimit = steps;

  //TODO - write upper limit to EEPROM?


  // Ok has been hit, check if retest is needed or proceed.
  // Write screen
  clearLCD();
  lcd.print(F("Bag inflated in"));
  lcd.setCursor(0,1);
  lcd.print(steps);
  lcd.setCursor(5,1);
  lcd.print(F(" steps."));
  lcd.setCursor(0,2);
  lcd.print(F("Press OK to continue"));
  lcd.setCursor(0,3);
  lcd.print(F("Press CONF to retry"));

  // wait for button press
  // Check if ok to proceed or retest
  while (!okBtnFlag && !confBtnFlag) {
    handleBTN();
  }
}



//####################################################################
/*
ZERO POTENTIOMETER
*/
// calibrate volume to VT pot
void zeroVolumePot() {
  Serial.println("Zeroing potentiometer..");
  // Write screen
  clearLCD();
  lcd.print(F("*** ML increments ***"));
  lcd.setCursor (0, 1);
  lcd.print(F("Turn VT fully"));
  lcd.setCursor (0, 2);
  lcd.print(F("anticlockwise."));
  lcd.setCursor (0, 3);
  lcd.print(F("Press OK to continue"));

  // wait for ok
  while (true) {
    //Serial.println("WAITING for ZERO");
    handleBTN();

    //if ok pressed
    if (okBtnFlag){
      Serial.println("WAITING for ZERO...");
      //get pot reading
      int potVal = cleanRead(VOLUME_POT);
      Serial.print("Pot val: ");
      Serial.println(potVal);

      //check if pot is actually set to zero
      if (potVal == 0){
        Serial.println("Pot at zero.");

        //reset ok flag
        okBtnFlag = false;

        //if zero reached, break out
        break;
      }
      //reset ok flag
      okBtnFlag = false;
    }
  }
}


//####################################################################
/*
ML CALLIBRATE CONFIGURATION
*/
void mlConfig(){
  int newVol = 0;
  int currentVol = 0;

  //Enable stepper
  digitalWrite(STEPPER_ENABLE, HIGH);
  digitalWrite(STEPPER_DIR, STEPPER_DIR_DOWN);

  // Move arm down slowly using VT pot until 50ml water displaced and press ok, or limit switch hit
  lcdDis = true;
  while (true) {
    //display message once
    if (lcdDis){
      clearLCD();
      lcd.print(F("*** ML increments ***"));
      lcd.setCursor (0, 1);
      lcd.print(F("Turn VT until 50 ml "));
      lcd.setCursor (0, 2);
      lcd.print(F("is reached."));
      lcd.setCursor (0, 3);
      lcd.print(F("Press OK when done"));
      lcdDis = false;
    }

    lcdDis = true;

    //get the pot reading and move arm if 'OK' not pressed
    while(!okBtnFlag && !confBtnFlag){
      //get the new reading
      newVol = cleanRead(VOLUME_POT);

      //compare the values
      if (newVol > currentVol + 5) {
        //Serial.println("Greater >>>>>");
        //Serial.print("New pot val: ");
        //Serial.println(newVol);

        //get pot difference (amount to move)
        int diff = (newVol - currentVol) * 3;

        //relative steps
        for (int i = 0; i < diff; i++){
          //Serial.print("STEP: ");
          //Serial.println(i);
          //step
          slowStep(500);
          steps--;
          //Serial.print("TOTAL STEPS: ");
          //Serial.println(steps);
        }

        //assign new value
        currentVol = newVol;

        //if reached the limit switch break out
        if (limitActived){
          //mode = ERR;
          //errFlag = true;
          //don't reset limit here
          break;
        }
      }

      // display next step message if not done
      if (lcdDis){
        clearLCD();
        lcd.print(F("*** ML increments ***"));
        lcd.setCursor (0, 1);
        lcd.print(F("Set 50 ml increment?"));
        lcd.setCursor (0, 2);
        lcd.print(F("OK=Yes, CONF=No"));
        lcdDis = false;
      }

      //handle ok pressed
      handleBTN();
    }


    //handle saving the value if 'OK' pressed
    if (okBtnFlag){
      //reset flag
      okBtnFlag = false;

      //ok pressed, save array
      Serial.println("SAVING!");

      //save the step marker to array
      config.stepsToVolume[calibIndex] = (int)steps;
      //increment index for
      calibIndex++;

      //view array
      for (int i = 0; i < STEP_TO_VOLUME_INCREMENTS; i++){
        //show the value for each
        Serial.print(config.stepsToVolume[i]);
        Serial.print(" ,");
      }

    //handle 'CONF' pressed
    }else if (confBtnFlag){
      // dont reset flag here!
      //confBtnFlag = false;
      //conf pressed, try again
      Serial.println("Returning....");
      //disable stepper
      digitalWrite(STEPPER_ENABLE, LOW);
      //break;
    }

    //raise flag if we are at the end of the array
    if(calibIndex == STEP_TO_VOLUME_INCREMENTS){
      //calibration finished
      calibDone = true;
    }

    //return to set ML_CONFIG
    break;
  }

  //disable stepper
  digitalWrite(STEPPER_ENABLE, LOW);
}


//####################################################################


/**********************************************************************
* SETUP
**********************************************************************/
void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  //inputs
  pinMode(OK_BTN_PIN, INPUT_PULLUP);
  pinMode(CONFIG_BTN_PIN, INPUT_PULLUP);
  pinMode(LIMIT_SWITCH_PIN, INPUT_PULLUP);
  pinMode(START_BTN_PIN, INPUT_PULLUP);

  //ISRs
  attachInterrupt(digitalPinToInterrupt(LIMIT_SWITCH_PIN), limitTriggered_ISR, FALLING);
  attachInterrupt(digitalPinToInterrupt(START_BTN_PIN), startTriggered_ISR, FALLING);

  //outputs
  pinMode(STEPPER_STEP, OUTPUT);
  pinMode(STEPPER_DIR, OUTPUT);
  pinMode(STEPPER_ENABLE, OUTPUT);
  pinMode(START_LED_PIN, OUTPUT);
  pinMode(BUZZER, OUTPUT);

  //Initial states
  digitalWrite(STEPPER_STEP, LOW);
  digitalWrite(STEPPER_DIR, STEPPER_DIR_DOWN);
  digitalWrite(STEPPER_ENABLE, LOW);
  digitalWrite(START_LED_PIN, LOW);
  digitalWrite(BUZZER, LOW);
  
  //initialise
  loadConfig();
  lcdInit();
  clearLCD();

  //Standby mode
  mode = STANDBY;

  //display msg
  lcdDis = true;
}



/**********************************************************************
* MAIN LOOP
**********************************************************************/
void loop() {
  if(limitActived){
    //Serial.println("Ready...");
    limitActived = false;
  }

  /*
  SWITCH MODES
  */
  switch (mode) {

    /*
    WAITING MODE
    */
    // case WAIT:
    //   clearLCD();
    //   lcd.print(F("*** WAITING ..... ***"));
    //   lcd.setCursor (0, 1);
    //   lcd.print(F("                     "));
    //   lcd.setCursor (0, 2);
    //   lcd.print(F("Press OK when ready."));

    //   Serial.println("Ready...");

    //   //wait for system to be ready and ok press
    //   while(true){
    //     handleBTN();
    //     if (okBtnFlag) {
    //       okBtnFlag = false;
    //       mode = RESET_ARM;
    //       break;
    //     }
    //   }
    // break;


    // /*
    // RESET MODE
    // */
    // case RESET_ARM:
    //   //reset steps
    //   Serial.println("Reset Mode");
    //   zeroArm();

    //   //increment mode to next state
    //   mode = STANDBY;
    //   lcdDis = true;


    //   break;



    /*
    STANDBY MODE
    */
    case STANDBY:
      //Serial.println("Standby Mode");
      if (lcdDis) {
        lcdDis = false;
        clearLCD();
        lcd.print(F("* Ventilator v0.90 *"));
        lcd.setCursor(0,2);
        lcd.print(F("Press OK to begin"));
        lcd.setCursor(0,3);
        lcd.print(F("Press CONFIG to cal"));
      }

      // Wait for user input
      handleBTN();

      //Config button pressed, go to config
      if (confBtnFlag) {
        confBtnFlag = false;
        //set config sequence start mode
        mode = VOL_CONFIG;
        lcdDis = true;
      }

      if (okBtnFlag) {
        okBtnFlag = false;
        //set ventilation mode
        mode = READY;
        lcdDis = true;
      }

      break;



    case VOL_CONFIG:
      /*
      CALIBRATE VOLUME OF BAG
      */
      Serial.println("Volume Config Mode");
      volConfig();
      // Recalibrate bag volume if config pressed
      if (okBtnFlag) {
        okBtnFlag = false;
        limitActived = false;
        mode = ML_CONFIG;
      }
      // Ok to proceed
      if (confBtnFlag) {
        confBtnFlag = false;
      }
      Serial.println("volume conf finished");
      break;




    /*
    CALLIBRATE ML
    */
    case ML_CONFIG:
      Serial.println("ML Config Mode.");
      //reset arm
      //zeroArm();
      //add user message to remove calib device
      clearLCD();
      lcd.print(F("* Ventilator v0.90 *"));
      lcd.setCursor(0,2);
      lcd.print(F("Attach calib. device"));
      lcd.setCursor(0,3);
      lcd.print(F("Press OK "));

      while(!okBtnFlag){
        handleBTN();
      }
      okBtnFlag = false;

      //reset flag
      limitActived = false;
      Serial.println("Resetting volume array.....");
      //reset volume increment array
      for (int i = 0; i < STEP_TO_VOLUME_INCREMENTS; i++){
        config.stepsToVolume[i] = (int)0;
        //Serial.println(i);
      }
      //reset calibration index
      calibIndex = 0;

      //add the first value to the array
      config.stepsToVolume[0] = config.stepsUpperLimit;
      calibIndex++;

      //while limit not reached
      while (!calibDone && !limitActived){
        //first zero pot by user
        zeroVolumePot();
        // Ok to proceed
        okBtnFlag = false;
        Serial.println("starting ML config");
        //turn pot until 50 ml
        mlConfig();
        //break out if hit bottom and calibration incomplete
        //shouldn't happen but just in case
        if (limitActived && !calibDone){
          Serial.println("LIMIT!!!");
          limitActived = false;
          break;
        }
        //if conf button pressed, incorrect calibration increment
        if(confBtnFlag){
          //reset flag
          confBtnFlag = false;
          //return to last valid increment
          resetToLast(config.stepsToVolume[calibIndex-1]);
        }
        //??
        limitActived = false;
      }
      //Check calibration complete
      if (calibDone){
        //save to EEPROM
        saveConfig();

        //user message to remove calibration device
        clearLCD();
        lcd.print(F("* Ventilator v0.90 *"));
        lcd.setCursor(0,2);
        lcd.print(F("Remove calib device!"));
        lcd.setCursor(0,3);
        lcd.print(F("Press OK "));

        while(!okBtnFlag){
          handleBTN();
        }
        okBtnFlag = false;


        //display next message
        lcdDis = true;
        //enter ready mode
        mode = READY;
      }

      break;





    /*
    READY TO START
    */
    case READY:
      //Serial.println("Ready Mode");

      if(lcdDis){
        //zero arm position
        zeroArm();

        //print message
        clearLCD();
        lcd.print(F("***** READY!! ******"));
        lcd.setCursor (0, 1);
        lcd.print(F("                    "));
        lcd.setCursor (0, 2);
        lcd.print(F("Press START to start"));
        lcd.setCursor (0, 3);
        lcd.print(F("Press OK to set"));
        

        //reset flag
        lcdDis = false;
      }
      
      //this small delay is required otherwise
      //pressing start fails
      //TODO - find out why???
      delay(10);

      //read buttons
      handleBTN();

      //check if OK pressed
      if(okBtnFlag){
        //enter user settings for volume, BPM, IE
        mode = SET;
        okBtnFlag = false;
      }
      
      break;

    
    
    /*
    SET LEVELS REQUIRED BEFORE START
    */
    case SET:

      Serial.println("Setting...");

      //wait for values to be set
      setValues();

      //print message
      clearLCD();
      lcd.print(F("***** READY!! ******"));
      lcd.setCursor (0, 1);
      lcd.print(F("                    "));
      lcd.setCursor (0, 2);
      lcd.print(F("Press START to start"));
      lcd.setCursor (0, 3);
      lcd.print(F("Press OK to set"));

      //change to run
      mode = READY;

      

      break;




    /*
    STARTED
    */
    case RUNNING:

      //Serial.println("Running...");

      //set the global breath settings (Volume, BPM, and I/E)
      // NOTE: pot values read and updated here
      handleSettings();

      //update screen
      handleScreen();

      //check buttons
      //probably don't need this..
      handleBTN();
     
      // Take one breath
      breath();      

      break;
      



    /*
    ERROR MODE
    */
    case ERR:
      Serial.println("ERROR!!!!!");

      //zero arm and wait for reset
      if (errFlag == true){
        zeroArm();
        errFlag = false;
      }

      break;



    default:
      Serial.println("default");
  }


}
