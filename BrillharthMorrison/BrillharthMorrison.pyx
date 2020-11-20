import cython
cimport cython
from math import trunc
import numpy as np
cimport numpy as np
from printing import *
# from gmpy2 import powmod
import gmpy2
from itertools import groupby, combinations, chain
from collections import Counter as cntr
from numba import jit

cdef class CFF:
    cdef:
        int n
        list factorBase
        list a
        list b
        list b_sq
        list result
#        np.ndarray[np.int_t, ndim = 2] matrix
#         short int[:] factorBase
#         short int[:] result


    def __init__(self, int n):
        self.n = n
        self.factorBase = []
        self.a = []
        self.b = []
        self.b.append(1)
        self.b_sq = []
#         self.factorBase = []
        self.result = []

    cdef list isCorrect(self):
        cdef list result = []
        #print(self.n % 2)
        if (self.n % 2 == 0):
            #print('first if +')
            result = [False, 'n % 2 != 0']
        elif self.n <= 16:
            result = [False, 'n <= 16']
        elif np.float(gmpy2.sqrt(self.n).is_integer()):
            result = [False, 'целый корень']
        elif self.checkPrime():
            result =  [False, 'простое число']
        else:
            result = [True, 'ok']
        return result

    cdef bint checkPrime(self):
        cdef bint result
        if len(self.factorizeNumber(self.n)) == 1:
            result = True
        else:
            result = False
        return result

    cdef list factorizeNumber(self, short int number):
        cdef list result = []
        cdef short int i = 0
#         number = self.n
        if number < 0:
            result.append(-1)
            number = -number
        for i in range(2, int(np.floor((np.float(gmpy2.sqrt(number))) + 1))):
            while (number % i == 0):
                result.append(i)
                number //= i
        if number != 1:
            result.append(number)
        return result

    cdef int squareByModGMPY2(self, int a, int mod):
        return np.int(gmpy2.powmod(a, 2, mod))

    cdef np.int_t squareByModNP(self, int a, int mod):
        return np.int(np.pow(a, 2) % mod)

    cdef int squareByModPy(self, int a, int mod):
      result = ((a * a) % mod)
      if result >= 0.85 * mod:
          return result - mod
      else:
          return result
      #return ret - mod if ret >= 0.85 * mod else ret

    cdef void addToFactorBase(self, short int n):
        cdef list result = self.factorizeNumber(n)
        cdef short int i = 0
        for i in range(len(result)):
            self.factorBase.append(result[i])
#         return self.factorBase

    cdef list createVector(self, int n, list factorBase):
        cdef list pos = []
        cdef list result = [0] * len(factorBase)
        cdef list factors = self.factorizeNumber(n)
        cdef factorsAndCounts = cntr(factors)
        cdef short int i = 0
        #cdef:
        #  list pos = []
        #  list result = [0] * len(factorBase)
        #  list coef = self.factorizeNumber(n)
        #  coef_and_count = cntr(coef)
        #  short int i = 0

        for i in range(len(factors)):
            pos.append(factorBase.index(factors[i]))
        pos = list(set(pos))
        for i in range(len(pos)):
            result[pos[i]] = (factorsAndCounts[factorBase[pos[i]]]) % 2
        return result

    cdef list createMatrix(self):
        cdef list result = []
        cdef short int i = 0
        for i in range(len(self.b_sq)):
            result.append(self.createVector(self.b_sq[i], self.factorBase))
        return result

    #@jit
    cdef list solveSystem(self, np.ndarray[np.int_t, ndim = 2] matrix):
        cdef list result = []
        cdef list idxs = self.findIndexes(matrix)
        cdef list solutions = []
        cdef list realSolutions = []
        cdef short int i = 0
        cdef int X
        cdef int Y
        cdef int solution1
        cdef int solution2
        for i in range(len(idxs)):
            X = self.calculateX(idxs[i], self.b, self.n)
            Y = self.calculateY(idxs[i], self.b_sq)
            solution1 = np.int(np.gcd(X - Y, self.n))
            solution2 = self.n / solution1#np.int(np.gcd(X + Y, self.n))
            #print(solution1, solution2, np.int(np.gcd(X - Y, self.n)), np.int(np.gcd(X + Y, self.n)))
            solutions.append([solution1, solution2])
            print('X = {0}, Y = {1}, НОД(X + Y, n) = {2}, НОД(X - Y, n) = {3}'.format(X, Y, solution2, solution1))

        for i in range(len(solutions)):
            if(
              ((solutions[i][0] == 1) or (solutions[i][0] == self.n)) or
              ((solutions[i][1] == 1) or (solutions[i][1] == self.n)) or
              (solutions[i][0] % self.n == solutions[i][1] % self.n)
            ):
                continue
            else:
                realSolutions.append(solutions[i])
        result = [element for element, empty in groupby(realSolutions)]
        return result

    cdef list findIndexes(self, np.ndarray[np.int_t, ndim = 2] matrix):
        #print('find indexes')
        cdef list indexes = []
        cdef short int i = 0
        cdef short int j = 0
        cdef list vectors = []
        cdef list tuples
        for i in range(2, len(matrix) + 1):
            tuples = list(combinations(matrix, i))
            #print('tuples ok')
            for j in range(len(tuples)):
                vectors = []
                #print('before if')
                if self.isEqual(tuples[j]):
                    #print('inside if')
                    for x in range(len(tuples[j])):
                        for y in range(len(matrix)):
                            if np.array_equal(matrix[y], tuples[j][x]):
                                vectors.append(y)
                    print('Проверяем комбинацию векторов №: {0}'.format(sorted(set(vectors))))
#                    print(sorted(set(vectors)))
                    indexes.append(sorted(list(set(vectors))))

        return [element for element, empty in groupby(indexes)]

    cdef void resultIsEmpty(self, list result):
        cdef string = ''
        if len(result) != 0:
            string = 'Разложение : {0} = {1} * {2}'.format(self.n, result[0][0], result[0][1])
        else:
            string = 'Нету комбинаций, удовлетворяющих условия'
        print(string)

    def isEqual(self, *tuple) -> bint:
        cdef bint is_equal
        cdef np.ndarray[np.float_t, ndim = 1] zeros = np.zeros(len(tuple[0][0]))
        cdef list result = []
        cdef short int i = 0
        cdef short int j = 0
        for i in range(len(tuple)):
            for j in range(len(tuple[i])):
                result.append(tuple[i][j])
        is_equal = np.array_equal(sum(result) % 2, zeros)
        return is_equal

    cdef int calculateX(self, list indexes, list b, int n):
        #cdef int result
        cdef long long int result = 1
        cdef short int i = 0
        for i in range(len(indexes)):
            result *= b[indexes[i] + 1]
        result %= n
        return result


    cdef int calculateY(self, list indexes, list b_sq):
        cdef long long int result = 1
        cdef list factors = []
        cdef short int i = 0
        for i in range(len(indexes)):
            factors.append(self.factorizeNumber(b_sq[indexes[i]]))
        coefs = list(chain(*factors))
        for i in range(len(coefs)):
            result *= coefs[i]
        return trunc(np.float(gmpy2.sqrt(result)))

#     def do_first_iteration, second_iteration, do_iterations

    cdef void printSep(self):
      print('#' * 100)
      print('#' * 100)

    cdef void doFirstIteration(self, np.int_t v_p, np.float_t alpha_0, np.int_t a_p, np.int_t u_p, np.int_t b_sq):
      cdef np.ndarray[np.int_t, ndim = 2] matrix
      printIterationInfo(0, v_p, alpha_0, a_p, u_p)
      self.b.append(a_p)
      self.b_sq.append(b_sq)
      self.a.append(a_p)


      self.addToFactorBase(b_sq)
      self.factorBase = sorted(list(set(self.factorBase)))

      print('a = {0}, b = {1}, b^2 = {2}, B = {3}'.format(self.a, self.b[1 :], self.b_sq, self.factorBase))

      matrix = np.array(self.createMatrix())
      printM(matrix, self.b_sq)
      result = self.solveSystem(matrix)
      self.resultIsEmpty(result)
      self.printSep()

    cdef tuple doSecondIteration(self, np.int_t v_p, np.float_t alpha_0, np.int_t a_p, np.int_t u_p, np.int_t b_sq):
      cdef np.ndarray[np.int_t, ndim = 2] matrix
      v = np.int((self.n - u_p * u_p) / v_p)
      alpha = np.round((alpha_0 + u_p) / v, 2)
      a = trunc(alpha)
      u = a * v - u_p
      printIterationInfo(1, v, alpha, a, u)
      b = self.b[1] * a + self.b[0]
      b_sq = self.squareByModPy(b, self.n)

      self.a.append(a)
      self.b.append(b)
      self.b_sq.append(b_sq)

      self.addToFactorBase(b_sq)
      self.factorBase = sorted(list(set(self.factorBase)))

      print('a = {0}, b = {1}, b^2 = {2}, B = {3}'.format(self.a, self.b[1 :], self.b_sq, self.factorBase))

      v_p = v
      a_p = a
      u_p = u
      matrix = np.array(self.createMatrix())
      printM(matrix, self.b_sq)
      result = self.solveSystem(matrix)
      self.resultIsEmpty(result)
      self.printSep()
      return result, v_p, a_p, u_p


    cdef doOtherIterations(self, np.float_t alpha_0, list result, np.int_t v_p, np.int_t a_p, np.int_t u_p):
      cdef short int i = 2
      while len(result) == 0:
          v = np.int((self.n - u_p * u_p) / v_p)
          alpha = np.round((alpha_0 + u_p) / v, 5)
          a = trunc(alpha)
          u = a * v - u_p
          printIterationInfo(i, v, alpha, a, u)
          b = self.b[i] * a + self.b[i - 1]
          b_sq = self.squareByModPy(b, self.n)

          self.a.append(a)
          self.b.append(b % self.n)
          self.b_sq.append(b_sq)

          self.addToFactorBase(b_sq)
          self.factorBase = sorted(list(set(self.factorBase)))

          print('a = {0}, b = {1}, b^2 = {2}, B = {3}'.format(self.a, self.b[1 :], self.b_sq, self.factorBase))

          a_p = a
          u_p = u
          v_p = v
          matrix = np.array(self.createMatrix())
          printM(matrix, self.b_sq)
          result = self.solveSystem(matrix)
          self.resultIsEmpty(result)
          self.printSep()
          i += 1


    cpdef void doAllStuff(self):
        cdef list correct = self.isCorrect()
        cdef np.int_t v_p = 1
        cdef np.float_t alpha_0 = np.round(np.float(np.float(gmpy2.sqrt(self.n))), 5)
        cdef np.int_t a_p = trunc(alpha_0)
        cdef np.int_t u_p = a_p
        cdef np.int_t b_sq = self.squareByModPy(a_p, self.n)
        cdef short int i = 2
        cdef list result = []
        cdef np.ndarray[np.int_t, ndim = 2] matrix
        if correct[0] == True:

            self.doFirstIteration(v_p, alpha_0, a_p, u_p, b_sq)
            result, v_p, a_p, u_p = self.doSecondIteration(v_p, alpha_0, a_p, u_p, b_sq)
            #print(result)
            self.doOtherIterations(alpha_0, result, v_p, a_p, u_p)
        else:
            print(correct[1])

cpdef test(a):
    printing(a)
