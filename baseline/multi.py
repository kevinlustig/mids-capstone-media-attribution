from classes.RangelandModeler import RangelandModeler
from classes.MultiTrainer import MultiTrainer

def dist_init(gpu,args):
    modeler = RangelandModeler()
    modeler.dist_init(gpu,args)

def main():
    
    mt = MultiTrainer(dist_init,'54.219.219.118')
    mt.spawn()

if __name__ == '__main__':
    main()



