from sklearn import tree
import numpy as np
import pickle
import sys

class DT():

    def __init__(self,fnames, dtname, maxDepth, dotname, notes):
        self.fnames = fnames # a list of strings
        self.dtname = dtname # string
        self.maxDepth = maxDepth # integer
        self.dotname = dotname # string
        self.notes = notes # string 
    
    def generate_and_save(self, fnames=None, dtname="default", maxDepth=3, dotName="default"):
        labels = []
        samples = []

        try:
            # a list of matrices
            data = []
            for fname in fnames:
                data.append( np.genfromtxt(fname, delimiter=",") ) # CSV only!

            print "Loaded data from csv"
        except Exception, err:
            print "Error loading csv file."
            print err

        try:
            sampleList = []
            for d in data:
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
            with open(dotname+".dot", 'w') as f:
                f = tree.export_graphviz(clf, out_file=f)

            pickle.dump(clf, open(dtname+".pkl","wb"))
        except Exception as e:
            print "Classifier or pickle error."
            print e

    def save_notes(self,dtname, dotname, notes):
        try:
            f = open("notes_for_classifier_"+dtname+".pkl_"+dotname+".dot_.txt","w")
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
    except Exception as e:
        print "Error using DTGenerator."
        print e
 
    print fnames
    print dtname
    print notes

    myDt = DT(fnames, dtname, maxDepth, dotname, notes)
    myDt.generate_and_save(fnames, dtname, maxDepth, dotname)
    myDt.save_notes(dtname, dotname, notes)
