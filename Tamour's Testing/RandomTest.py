import random

# setting Q to a very large prime number
Q = 23740629843760239486723


def encrypt(x, n_share=3):
    r"""Returns a tuple containg n_share number of shares
    obtained after encrypting the value x."""

    shares = list()
    for i in range(n_share - 1):
        shares.append(random.randint(0, Q))
    shares.append(Q - (sum(shares) % Q) + x)
    return tuple(shares)


print("Shares: " + str(encrypt(3)))

def decrypt(shares):
    r"""Returns a value obtained by decrypting the shares."""

    return sum(shares) % Q


print("Value after decrypting: " + str(decrypt(encrypt(3))))