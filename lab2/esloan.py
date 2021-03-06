def ask_int(question):
    print(question)
    return int(input())

def ask_boolean(question):
    print(question)
    res = input()
    if (res.lower() == "yes" or res.lower() == "y"):
        return True
    elif (res.lower() == "no" or res.lower() == "n"):
        return False
    else:
        return ask_boolean(question)

def qualify():

    res = ask_boolean("Do already you have a mortgage?")
    if (res):
        return False
    
    house_val = ask_int("What is the value of your property?")
    income = ask_int("What is your current salary?")

    thresh = house_val/20
    if (thresh > income):
        return False
    
    front = ask_int("How much do you intend to give upfront?")
    if (thresh/2 > front):
        return False
    
    spend = ask_int("How much do you spend per month?")
    if (spend > income/4):
        return False
    
    return True


if (qualify()):
    print("Congratulations! You have qualified for a mortgage!")
else:
    print("Unfortunately you do not meet our requirements")