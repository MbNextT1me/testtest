#include <SoftwareSerial.h>

// Определение пинов для сдвигового регистра
#define DATA_PIN 8
#define LACTH_PIN 9
#define CLOCK_PIN 10

// Определение пинов для SoftwareSerial (общение ардуинок)

#define RX_PIN 2  // (Пин для получения данных) 
#define TX_PIN 3  // (Пин для отправки данных) 

SoftwareSerial Slave(RX_PIN, TX_PIN);	  // rx, tx

bool mess_input = true;  // Флаг, чтобы вывводить строку "Enter number: " только 1 раз перед вводом числа
long num = 0;

uint8_t nums[] = { 0b11101110, 0b10000010, 0b11011100, 0b11010110, 0b10110010, 0b01110110, 0b01111110, 0b11000010, 0b11111110, 0b11110110 };  // Массив чисел от 0 до 9 в виде данных для сдвигового регистра

void setup()
{
    // Определение режима работы пинов
  pinMode(LACTH_PIN, OUTPUT);
  pinMode(CLOCK_PIN, OUTPUT);
  pinMode(DATA_PIN, OUTPUT);

  pinMode(RX_PIN, INPUT);
  pinMode(TX_PIN, OUTPUT);
  
  Serial.begin(9600);  // Серийный порт для общения с компьютером
  Slave.begin(9600);  // Серийный порт для общения с другой ардуино
}

void loop()
{
  if (mess_input)
  {
    Serial.print("Enter number: ");  // Просим ввести число
    mess_input = false;  // Теперь не вывводим это сообщение, пока пользователь число не введет
  }

  if (Serial.available())  // Если пользователь что-то ввел
  {
    num = Serial.parseInt();  // парсим число
    Serial.println(num);  // Вывводим на компьютер

    Slave.print(num % 10);  // Отдаем другой ардуине вторую цифру числа

    num = num / 10;  // Достаем первую цифру (откидываем вторую)

    show_digit(num % 10);  // Показываем первую цифру числа
    
    while (!Slave.available()){}	// Ждем пока вторая ардуино напечатает и ответит нам
    Slave.read();

    delay(800);	// Чтобы успеть увидеть выввод
    
    mess_input = true;  // Теперь надо выввести сообщение о вводе нового числа
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
