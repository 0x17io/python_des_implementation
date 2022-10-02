##########################################################################################################
# Name: Jose Ruben Espinoza                                                                              #
# Summary: Pythonic implementation of DES algorithm. Both encryption and decryption are studied here.    #
# Sources: Ideas were developed using Chapter 06 of your textbook: Cryptography and Network Security,    #
#          by William Stallings. Permutation arrays are standards.                                       #
##########################################################################################################

# Import necessary libraries
import numpy as np

##########################################################################################################
#                           Begin DES Portion of Code                                                    #
##########################################################################################################
# Define initial and final permutations.
initial_permutation = [58, 50, 42, 34, 26, 18, 10, 2,
                       60, 52, 44, 36, 28, 20, 12, 4,
                       62, 54, 46, 38, 30, 22, 14, 6,
                       64, 56, 48, 40, 32, 24, 16, 8,
                       57, 49, 41, 33, 25, 17, 9, 1,
                       59, 51, 43, 35, 27, 19, 11, 3,
                       61, 53, 45, 37, 29, 21, 13, 5,
                       63, 55, 47, 39, 31, 23, 15, 7]
final_permutation = [40, 8, 48, 16, 56, 24, 64, 32,
                     39, 7, 47, 15, 55, 23, 63, 31,
                     38, 6, 46, 14, 54, 22, 62, 30,
                     37, 5, 45, 13, 53, 21, 61, 29,
                     36, 4, 44, 12, 52, 20, 60, 28,
                     35, 3, 43, 11, 51, 19, 59, 27,
                     34, 2, 42, 10, 50, 18, 58, 26,
                     33, 1, 41, 9, 49, 17, 57, 25]

# Recall that dictionaries are fast, we use them to define our s_boxes
# note that these boxes are gotten from our textbook and DES documentation
SBox = {0: [[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
            [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
            [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
            [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],
        1: [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
            [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
            [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
            [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],
        2: [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
            [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
            [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
            [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],
        3: [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
            [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
            [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
            [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],
        4: [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
            [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
            [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
            [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],
        5: [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
            [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
            [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
            [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],
        6: [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
            [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
            [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
            [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],
        7: [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
            [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
            [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
            [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]]
        }


# Straight permutation table.
string_permutation_table = [16, 7, 20, 21, 29, 12, 28, 17,
                            1, 15, 23, 26, 5, 18, 31, 10,
                            2, 8, 24, 14, 32, 27, 3, 9,
                            19, 13, 30, 6, 22, 11, 4, 25]

# Expansion D-Box
expansion_d_box = [32, 1, 2, 3, 4, 5,
                   4, 5, 6, 7, 8, 9,
                   8, 9, 10, 11, 12, 13,
                   12, 13, 14, 15, 16, 17,
                   16, 17, 18, 19, 20, 21,
                   20, 21, 22, 23, 24, 25,
                   24, 25, 26, 27, 28, 29,
                   28, 29, 30, 31, 32, 1]


# For key generation (pc1)
parity_bit_drop_table = [57, 49, 41, 33, 25, 17, 9, 1,
                         58, 50, 42, 34, 26, 18, 10, 2,
                         59, 51, 43, 35, 27, 19, 11, 3,
                         60, 52, 44, 36, 63, 55, 47, 39,
                         31, 23, 15, 7, 62, 54, 46, 38,
                         30, 22, 14, 6, 61, 53, 45, 37,
                         29, 21, 13, 5, 28, 20, 12, 4]

# For key generation (pc2)
key_compression_table = [14, 17, 11, 24, 1, 5, 3, 28,
                         15, 6, 21, 10, 23, 19, 12, 4,
                         26, 8, 16, 7, 27, 20, 13, 2,
                         41, 52, 31, 37, 47, 55, 30, 40,
                         51, 45, 33, 48, 44, 49, 39, 56,
                         34, 53, 46, 42, 50, 36, 29, 32]


def split_message(message):
    """
    Splits a given bit message black in half.
    :param message: 64 bit message
    :return: Tuple: (left_side, right_side)
    """
    return message[0:32], message[32:65]


def permute_bits(bits, permutation):
    """
    Allows for permutation of bits given a table. Note that we utilize array manipulations for ease of development.
    :param bits: binary bits as strings
    :param permutation: Permutation table.
    :return: resulting permutation
    """
    returning_bits = list(bits)
    returning_arr = []
    for idx, value in enumerate(permutation):
        returning_arr.append(returning_bits[value - 1])
    return ''.join(returning_arr)

def key_generation(key):
    """
    Generate all keys using the DES standard.
    :param key: 64 bit key
    :return: Dictionary of binary keys.
    """
    key_dict = {}
    permuted_bits = permute_bits(key, parity_bit_drop_table)

    for num in range(1, 17):
        bit_split = (permuted_bits[0:28], permuted_bits[28:])
        if num == 1 or num == 2 or num == 9 or num == 16:
            shifted = (np.roll([bit for bit in bit_split[0]], -1), np.roll([bit for bit in bit_split[1]], -1))
            shifted = ''.join(shifted[0]) + ''.join(shifted[1])
        else:
            shifted = (np.roll([bit for bit in bit_split[0]], -2), np.roll([bit for bit in bit_split[1]], -2))
            shifted = ''.join(shifted[0]) + ''.join(shifted[1])
        key_dict[num] = permute_bits(shifted, key_compression_table)
        permuted_bits = shifted
    return key_dict

def des_function(bits32, key):
    """
    Inner workings of DES function.
    :param bits32: 32 bits
    :param key: 48 bits
    :return: Permuted bits.
    """
    post_expansion = permute_bits(bits32, expansion_d_box)
    xor_operation = bin(int(post_expansion, 2) ^ int(key, 2))[2:].zfill(48)

    if len(xor_operation) != 48:
        # On here as a safety.
        print("Something went wrong.")
    else:
        pieces_of_6_bits = [xor_operation[i: i+6] for i in range(0, len(xor_operation), 6)]
        post_s_boxes = []
        #print(pieces_of_6_bits)
        for idx, subset in enumerate(pieces_of_6_bits):
            row = int(subset[0] + subset[5], 2)
            column = int(subset[1:5], 2)
            post_s_boxes.append(bin(SBox[idx][row][column])[2:].zfill(4))
        s_box_num = ''.join(post_s_boxes)

    return permute_bits(s_box_num, string_permutation_table)

def des_main(message, key_dict, decrypt_vs_encrpty):
    """
    Main driver code for DES encryption.
    :param message: 64bit message black
    :param key_dict: keys 1-16, used for rounds
    :return: binary ciphertext
    """
    message = permute_bits(message, initial_permutation)
    # Feistel Cipher
    left_side, right_side = split_message(message)

    if decrypt_vs_encrpty == "encrypt":
        key_order = range(1, 17)
    else:
        key_order = list(reversed(range(1,17)))
    for rounds in key_order:
        post_function = des_function(right_side, key_dict[rounds])
        new_right = bin(int(left_side, 2) ^ int(post_function, 2))[2:].zfill(32)
        new_left = right_side
        right_side = new_right
        left_side = new_left


    return permute_bits(right_side+left_side, final_permutation)

def convert_to_64bit_blocks(message):
    """
    Returns blocks of 64 bits, padding of 0s if message length is not divisible by 64.
    :param message: Text message.
    :return: Array of 64 bit elements.
    """
    bits_of_64_conversion = [ord(c) for c in message]
    bits_of_64_conversion = [bin(num)[2:].zfill(8) for num in bits_of_64_conversion]
    bits_of_64_conversion = ''.join(bits_of_64_conversion)

    needed_bits = 64 - (len(bits_of_64_conversion) % 64)

    # Converting to list, since operations are more intuitive
    convert_2_list = list(bits_of_64_conversion)
    convert_2_list.extend(['0'] * needed_bits)

    bits_of_64_conversion = ''.join(convert_2_list)

    return [bits_of_64_conversion[i:i+64] for i in range(0, len(bits_of_64_conversion), 64)]
##########################################################################################################
#                           Ending DES Portion of Code                                                   #
##########################################################################################################


# Driver code
if __name__ == '__main__':

    key = '133457799BBCDFF1' # if this is changed make sure it's a 64 bit
    key_2_hex = [key[i:i+2] for i in range(0, len(key), 2)]
    key_2_binary = ''.join([bin(int(num, 16))[2:].zfill(8) for num in key_2_hex])

    key_dict = key_generation(key_2_binary)

    # Showcasing the encryption of a message.
    message_to_encrypt = "Hello world!"
    print("Original message: ", message_to_encrypt)
    message_to_encrypt = convert_to_64bit_blocks(message_to_encrypt)
    print("Original message as bits of 64: ", message_to_encrypt)

    encrypted_message = []
    for piece in message_to_encrypt:
        encrypted_message.append(des_main(piece, key_dict, "encrypt"))
    encrypted_message = ''.join(encrypted_message)

    print("Encrypted bits: ", encrypted_message)

    # Showcasing the decryption of a message.
    # First break into array of 64 bits
    breaking_message_64s = [encrypted_message[i:i + 64] for i in range(0, len(encrypted_message), 64)]
    print("Encrypted message as bits of 64: ", breaking_message_64s)
    print("\n\n")
    decrypted_message = []
    for piece in breaking_message_64s:
        decrypted_message.append(des_main(piece, key_dict, "decrypt"))

    decrypted_message = ''.join(decrypted_message)
    print("Decrypted message as bits: ", decrypted_message)

    decrypted_message = [chr(int(decrypted_message[i: i + 8], 2)) for i in range(0, len(decrypted_message), 8)
                        if chr(int(decrypted_message[i: i + 8], 2)) != '\x00']

    decrypted_message = ''.join(decrypted_message)
    print(decrypted_message)


