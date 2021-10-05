import myfunctions

print(myfunctions.sum(5,5))      # 10
print(myfunctions.subtract(5,2)) # 3
print(myfunctions.multiply(5,5)) # 25
print(myfunctions.divide(10,5))  # 2

print("Please enter a number: ")
number1 = int(input())

print("Please enter a second number: ")
number2 = int(input())

action = '0'

while (action != '1' and action != '2' and action != '3' and action != '4'):
    print("What would you like to do with these numbers?")
    print("1: Add")
    print("2: Subtract")
    print("3: Multiply")
    print("4: Divide")
    print()
    action = input()

if action == "1":
    print(myfunctions.sum(number1, number2))
elif action == "2":
    print(myfunctions.subtract(number1, number2))
elif action == "3":
    print(myfunctions.multiply(number1, number2))
elif action == "4":
    print(myfunctions.divide(number1, number2))
