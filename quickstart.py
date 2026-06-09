from pyrespect_freq import ReSpect

solver = ReSpect()
solver.fit("tests/test1.dat")
solver.save(which="full", path="output/")
solver.plot(which="base")