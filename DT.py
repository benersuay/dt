from sklearn import tree
import numpy as np
import pickle
import sys

class DT():

    def __init__(self,fnames, dtdir, dtname, maxDepth, dotname, notes):
        self.fnames = fnames # a list of strings
        self.dtdir = dtdir # string
        self.dtname = dtname # string
        self.maxDepth = maxDepth # integer
        self.dotname = dotname # string
        self.notes = notes # string 
    
    def generate_and_save(self, fnames=None, dtdir=".", dtname="default", maxDepth=3, dotname="default"):
        labels = []
        samples = []

        try:
            # a list of matrices
            data = []
            for fname in fnames:
                matrix = np.genfromtxt(fname, delimiter=",") # CSV only!
                
                # Handle the exception for single sample files (that is, data files with only one line)
                # (numpy returns a list instead of a 2D array for 1 line files, which causes trouble in data shape
                if( len( matrix.shape ) == 1 ):
                    matrix = np.reshape( matrix, (1, len(matrix)) )
                    
                data.append( matrix ) 

            print "Loaded data from csv"
        except Exception, err:
            print "Error loading csv file."
            print err

        try:
            sampleList = []
            for dIdx, d in enumerate(data):
                print "Reading file: "+str(fnames[dIdx])
                # multiple entry (i.e. sample) file
                labels.extend(d[:,0])
                sampleList.append(d[:,1:])
                    
            samples = np.vstack(tuple(sampleList)) # convert sample list to a tuple
        except Exception as e:
            print "Error reshaping data."
            print e


        # print samples
        # print labels

        try:
            clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=maxDepth)
            clf = clf.fit(samples, labels)
            with open(dtdir+'/'+dotname+".dot", 'w') as f:
                f = tree.export_graphviz(clf, out_file=f)

            pickle.dump(clf, open(dtdir+'/'+dtname+".pkl","wb"))
        except Exception as e:
            print "Classifier or pickle error."
            print e

    def save_notes(self,dtdir, dtname, dotname, notes):
        try:
            f = open(dtdir+'/'+"notes_for_classifier_"+dtname+".pkl_"+dotname+".dot_.txt","w")
            f.write(notes)
            f.close()
        except Exception as e:
            print "Error saving notes."
            print e

    def load(self):
        pass

    def predict(self):
        pass

if __name__ == "__main__" : 
    fnames = []
    dtname = ""
    dtdir = ""
    notes = ""
    try:
        for aIdx, a in enumerate(sys.argv):
            if a[-4:] == ".csv":
                fnames.append(a)

            if a == "--name":
                dtname = sys.argv[aIdx + 1]

            if a == "--notes":
                notes = sys.argv[aIdx + 1]

            if a == "--dot":
                dotname = sys.argv[aIdx + 1]

            if a == "--depth":
                maxDepth = int(sys.argv[aIdx + 1])

            if a == "--dtdir":
                dtdir = sys.argv[aIdx + 1]
                
    except Exception as e:
        print "Error using DT class."
        print e
 
    print fnames
    print dtname
    print notes

    myDt = DT(fnames, dtdir, dtname, maxDepth, dotname, notes)
    myDt.generate_and_save(fnames, dtdir, dtname, maxDepth, dotname)
    myDt.save_notes(dtdir, dtname, dotname, notes)
