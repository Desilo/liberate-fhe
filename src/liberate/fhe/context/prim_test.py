import random


def MillerRabinPrimalityTest(number, rounds=10):
    # If the input is an even number, return immediately with False.
    if number == 2:
        return True
    elif number == 1 or number % 2 == 0:
        return False

    # First we want to express n as : 2^s * r ( were r is odd )

    # The odd part of the number
    oddPartOfNumber = number - 1

    # The number of time that the number is divided by two
    timesTwoDividNumber = 0

    # while r is even divid by 2 to find the odd part
    while oddPartOfNumber % 2 == 0:
        oddPartOfNumber = oddPartOfNumber / 2
        timesTwoDividNumber = timesTwoDividNumber + 1

    # Make oddPartOfNumber integer.
    oddPartOfNumber = int(oddPartOfNumber)

    # Since there are number that are cases of "strong liar" we need to check more than one number
    for time in range(rounds):
        # Choose "Good" random number
        while True:
            # Draw a RANDOM number in range of number ( Z_number )
            randomNumber = random.randint(2, number) - 1
            if randomNumber != 0 and randomNumber != 1:
                break

        # randomNumberWithPower = randomNumber^oddPartOfNumber mod number
        randomNumberWithPower = pow(randomNumber, oddPartOfNumber, number)

        # If random number is not 1 and not -1 ( in mod n )
        if (randomNumberWithPower != 1) and (
            randomNumberWithPower != number - 1
        ):
            # number of iteration
            iterationNumber = 1

            # While we can squre the number and the squered number is not -1 mod number
            while (iterationNumber <= timesTwoDividNumber - 1) and (
                randomNumberWithPower != number - 1
            ):
                # Squre the number
                randomNumberWithPower = pow(randomNumberWithPower, 2, number)

                # inc the number of iteration
                iterationNumber = iterationNumber + 1

            # If x != -1 mod number then it is because we did not find strong witnesses
            # hence 1 have more then two roots in mod n ==>
            # n is composite ==> return false for primality

            if randomNumberWithPower != (number - 1):
                return False

    # The number pass the tests ==> it is probably prime ==> return true for primality
    return True
