// Определение пинов для сдвигового регистра

#define DATA_PIN 8
#define LACTH_PIN 9
#define CLOCK_PIN 10

uint8_t nums[] = { 0b11101110, 0b10000010, 0b11011100, 0b11010110, 0b10110010, 0b01110110, 0b01111110, 0b11000010, 0b11111110, 0b11110110 };  // Массив чисел от 0 до 9 в виде данных для сдвигового регистра

int num = 0;

void setup()
{
  // Определение режима работы пинов
  pinMode(LACTH_PIN, OUTPUT);
  pinMode(CLOCK_PIN, OUTPUT);
  pinMode(DATA_PIN, OUTPUT);
 
  Serial.begin(9600);
}

void loop()
{
  if (Serial.available())  // Если нам пришла цифра от главной ардуины
  {
    num = Serial.parseInt();  // берем число

    show_digit(num % 10);  // Показываем цифру числа
    
    Serial.write("1");  // Говорим главной ардуине, что показали ее
  }
}

void show_digit(int8_t digit)
{
    if (digit < 0 || digit > 9)
        return;

  	digitalWrite(LACTH_PIN, LOW);  // Говорим сдвиговому регистру, что будем вводить данные
    shiftOut(DATA_PIN, CLOCK_PIN, MSBFIRST, (255 - nums[digit]));  // Вводим данные
	  digitalWrite(LACTH_PIN, HIGH);  // Говорим сдвиговому регистру, что данные ввели
    
    return;
}
