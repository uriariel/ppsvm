import numpy as np
from seal import Plaintext, PublicKey, Ciphertext, CKKSEncoder, SEALContext, Encryptor, \
    Decryptor, Evaluator, SecretKey, RelinKeys, GaloisKeys, DoubleVector, EncryptionParameters, scheme_type, \
    CoeffModulus, KeyGenerator

from smart.constants import POLY_MODULUS_DEGREE, PRIME_SIZE_LIST, SCALE
from smart.seal_matrix import CipherMatrix, Matrix
from smart.seal_vector import Vector


class SealOps:
    @classmethod
    def with_env(cls):
        parms = EncryptionParameters(scheme_type.CKKS)
        parms.set_poly_modulus_degree(POLY_MODULUS_DEGREE)
        parms.set_coeff_modulus(CoeffModulus.Create(POLY_MODULUS_DEGREE, PRIME_SIZE_LIST))

        context = SEALContext.Create(parms)

        keygen = KeyGenerator(context)
        public_key = keygen.public_key()
        secret_key = keygen.secret_key()
        relin_keys = keygen.relin_keys()
        galois_keys = keygen.galois_keys()

        return cls(context=context, public_key=public_key, secret_key=secret_key, relin_keys=relin_keys,
                   galois_keys=galois_keys, poly_modulus_degree=POLY_MODULUS_DEGREE, scale=SCALE)

    def __init__(self, context: SEALContext, scale: float, poly_modulus_degree: int, public_key: PublicKey = None,
                 secret_key: SecretKey = None, relin_keys: RelinKeys = None, galois_keys: GaloisKeys = None):
        self.scale = scale
        self.context = context
        self.encoder = CKKSEncoder(context)
        self.evaluator = Evaluator(context)
        self.encryptor = Encryptor(context, public_key)
        self.decryptor = Decryptor(context, secret_key)
        self.relin_keys = relin_keys
        self.galois_keys = galois_keys
        self.poly_modulus_degree_log = np.log2(poly_modulus_degree)

    def encrypt(self, matrix: np.array):
        matrix = Matrix.from_numpy_array(array=matrix)
        cipher_matrix = CipherMatrix(rows=matrix.rows, cols=matrix.cols)

        for i in range(matrix.rows):
            encoded_row = Plaintext()
            self.encoder.encode(matrix[i], self.scale, encoded_row)
            self.encryptor.encrypt(encoded_row, cipher_matrix[i])

        return cipher_matrix

    def decrypt(self, cipher_matrix: CipherMatrix) -> Matrix:
        matrix = Matrix(rows=cipher_matrix.rows, cols=cipher_matrix.cols)

        for i in range(matrix.rows):
            row = Vector()
            encoded_row = Plaintext()
            self.decryptor.decrypt(cipher_matrix[i], encoded_row)
            self.encoder.decode(encoded_row, row)
            matrix[i] = row

        return matrix

    def add(self, matrix_a: CipherMatrix, matrix_b: CipherMatrix) -> CipherMatrix:
        self.validate_same_dimension(matrix_a, matrix_b)

        result_matrix = CipherMatrix(rows=matrix_a.rows, cols=matrix_a.cols)
        for i in range(matrix_a.rows):
            a_tag, b_tag = self.get_matched_scale_vectors(matrix_a[i], matrix_b[i])
            self.evaluator.add(a_tag, b_tag, result_matrix[i])

        return result_matrix

    def add_plain(self, matrix_a: CipherMatrix, matrix_b: np.array) -> CipherMatrix:
        matrix_b = Matrix.from_numpy_array(matrix_b)
        self.validate_same_dimension(matrix_a, matrix_b)

        result_matrix = CipherMatrix(rows=matrix_a.rows, cols=matrix_a.cols)

        for i in range(matrix_a.rows):
            row = matrix_b[i]
            encoded_row = Plaintext()
            self.encoder.encode(row, self.scale, encoded_row)
            self.evaluator.mod_switch_to_inplace(encoded_row, matrix_a[i].parms_id())
            self.evaluator.add_plain(matrix_a[i], encoded_row, result_matrix[i])

        return result_matrix

    def multiply_plain(self, matrix_a: CipherMatrix, matrix_b: np.array) -> CipherMatrix:
        matrix_b = Matrix.from_numpy_array(matrix_b)
        self.validate_same_dimension(matrix_a, matrix_b)

        result_matrix = CipherMatrix(rows=matrix_a.rows, cols=matrix_a.cols)

        for i in range(matrix_a.rows):
            row = matrix_b[i]
            encoded_row = Plaintext()
            self.encoder.encode(row, self.scale, encoded_row)
            self.evaluator.mod_switch_to_inplace(encoded_row, matrix_a[i].parms_id())
            self.evaluator.multiply_plain(matrix_a[i], encoded_row, result_matrix[i])

        return result_matrix

    def dot_vector(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        result = Ciphertext()

        self.evaluator.multiply(a, b, result)
        self.evaluator.relinearize_inplace(result, self.relin_keys)
        self.vector_sum_inplace(result)
        self.get_vector_first_element(result)
        self.evaluator.rescale_to_next_inplace(result)

        return result

    def dot_vector_with_plain(self, a: Ciphertext, b: DoubleVector) -> Ciphertext:
        result = Ciphertext()

        b_plain = Plaintext()
        self.encoder.encode(b, self.scale, b_plain)

        self.evaluator.multiply_plain(a, b_plain, result)
        self.vector_sum_inplace(result)
        self.get_vector_first_element(result)

        self.evaluator.rescale_to_next_inplace(result)

        return result

    def get_vector_range(self, vector_a: Ciphertext, i: int, j: int) -> Ciphertext:
        cipher_range = Ciphertext()

        one_and_zeros = DoubleVector([0.0 if x < i else 1.0 for x in range(j)])
        plain = Plaintext()
        self.encoder.encode(one_and_zeros, self.scale, plain)
        self.evaluator.mod_switch_to_inplace(plain, vector_a.parms_id())
        self.evaluator.multiply_plain(vector_a, plain, cipher_range)

        return cipher_range

    def dot_matrix_with_matrix_transpose(self, matrix_a: CipherMatrix, matrix_b: CipherMatrix):
        result_matrix = CipherMatrix(rows=matrix_a.rows, cols=matrix_a.cols)

        rows_a = matrix_a.rows
        cols_b = matrix_b.rows

        for i in range(rows_a):
            vector_dot_products = []
            zeros = Plaintext()

            for j in range(cols_b):
                vector_dot_products += [self.dot_vector(matrix_a[i], matrix_b[j])]

                if j == 0:
                    zero = DoubleVector()
                    self.encoder.encode(zero, vector_dot_products[j].scale(), zeros)
                    self.evaluator.mod_switch_to_inplace(zeros, vector_dot_products[j].parms_id())
                    self.evaluator.add_plain(vector_dot_products[j], zeros, result_matrix[i])
                else:
                    self.evaluator.rotate_vector_inplace(vector_dot_products[j], -j, self.galois_keys)
                    self.evaluator.add_inplace(result_matrix[i], vector_dot_products[j])

        for vec in result_matrix:
            self.evaluator.relinearize_inplace(vec, self.relin_keys)
            self.evaluator.rescale_to_next_inplace(vec)

        return result_matrix

    def dot_matrix_with_plain_matrix_transpose(self, matrix_a: CipherMatrix, matrix_b: np.array):
        matrix_b = Matrix.from_numpy_array(matrix_b)
        result_matrix = CipherMatrix(rows=matrix_a.rows, cols=matrix_a.cols)

        rows_a = matrix_a.rows
        cols_b = matrix_b.rows

        for i in range(rows_a):
            vector_dot_products = []
            zeros = Plaintext()

            for j in range(cols_b):
                vector_dot_products += [self.dot_vector_with_plain(matrix_a[i], matrix_b[j])]

                if j == 0:
                    zero = DoubleVector()
                    self.encoder.encode(zero, vector_dot_products[j].scale(), zeros)
                    self.evaluator.mod_switch_to_inplace(zeros, vector_dot_products[j].parms_id())
                    self.evaluator.add_plain(vector_dot_products[j], zeros, result_matrix[i])
                else:
                    self.evaluator.rotate_vector_inplace(vector_dot_products[j], -j, self.galois_keys)
                    self.evaluator.add_inplace(result_matrix[i], vector_dot_products[j])

        for vec in result_matrix:
            self.evaluator.relinearize_inplace(vec, self.relin_keys)
            self.evaluator.rescale_to_next_inplace(vec)

        return result_matrix

    @staticmethod
    def validate_same_dimension(matrix_a, matrix_b):
        if matrix_a.rows != matrix_b.rows or matrix_a.cols != matrix_b.cols:
            raise ArithmeticError("Matrices aren't of the same dimension")

    def vector_sum_inplace(self, cipher: Ciphertext):
        rotated = Ciphertext()

        for i in range(int(self.poly_modulus_degree_log - 1)):
            self.evaluator.rotate_vector(cipher, pow(2, i), self.galois_keys, rotated)
            self.evaluator.add_inplace(cipher, rotated)

    def get_vector_first_element(self, cipher: Ciphertext):
        one_and_zeros = DoubleVector([1.0])
        plain = Plaintext()
        self.encoder.encode(one_and_zeros, self.scale, plain)
        self.evaluator.multiply_plain_inplace(cipher, plain)

    def get_matched_scale_vectors(self, a: Ciphertext, b: Ciphertext) -> (Ciphertext, Ciphertext):
        a_tag = Ciphertext(a)
        b_tag = Ciphertext(b)

        a_index = self.context.get_context_data(a.parms_id()).chain_index()
        b_index = self.context.get_context_data(b.parms_id()).chain_index()

        # Changing the mod if required, else just setting the scale
        if a_index < b_index:
            self.evaluator.mod_switch_to_inplace(b_tag, a.parms_id())

        elif a_index > b_index:
            self.evaluator.mod_switch_to_inplace(a_tag, b.parms_id())

        a_tag.set_scale(self.scale)
        b_tag.set_scale(self.scale)

        return a_tag, b_tag
