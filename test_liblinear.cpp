#include <iostream>
#include <fstream>
#include "nms.hpp"
#include "linear.h"
#include "xmlreader.h"
#include <vector>
#include <string>
#include <boost/filesystem.hpp>


using namespace std;

int loadims(string infile, vector<string>& ims){
	ifstream fin(infile.c_str());
	while(!fin.eof()){
		string tmp;
		fin >> tmp;
		if(tmp==""){
			break;
		}
		ims.push_back(tmp);
	}
	return 0;
}

int predict(vector<string>& ims, string modelfile, string outputfile){
	namespace bf=boost::filesystem;	
	struct model* m;
	if((m=load_model(modelfile.c_str()))==0)
	{
			cout <<"can't open model file ";
			exit(1);
	}
	int nr_class = get_nr_class(m);
	assert(nr_class == 2);
	double* w = m->w;
	double bias = m->bias;
	int nr_feature = m->nr_feature;
	bias = bias*w[nr_feature];
	ofstream fo(outputfile.c_str());
	for(int i=0;i<ims.size();i++){
		bf::path im = ims[i];
		vector< vector<int> > boxes;
		vector<double> scores;
		string PureName = im.stem().string();
		cout << "processing ims: " << PureName << "(" << i+1 << "/" << ims.size() << ")" << endl;
		ifstream fin(im.string().c_str());
		int num, dim;
		fin >> num >> dim;
		assert(dim == nr_feature);
		for(int j=0;j<num;j++){
			
			string im_id;
			fin >> im_id;
			assert(im_id == PureName);
			
			vector<int> box(4,0);
			fin >> box[0] >> box[1] >> box[2] >> box[3];
			double s=0;
			//vector<float> ftr(dim,0);
			float ftr = 0;
			for(int k=0;k<dim;k++){
				fin >> ftr;
				s += ftr*w[k];
			}
			s += bias;
			scores.push_back(s);
			boxes.push_back(box);
			//double s = 0;
			//for(int
		}
		fin.close();
		vector<pair<vector<int>, double> > picked;
		nms(boxes, scores, picked, 0.3);
		for(int j=0;j<picked.size();j++){
			double s=picked[j].second;
			vector<int>& box = picked[j].first;
			fo << PureName << " " << s << " " << box[0] << " " << box[1] << " " << box[2] << " " << box[3] << endl;
		}
	}
	fo.close();
	return 0;
}

int main(int argc, char** argv){
	if(argc < 4) {
		cout << "Usage:" << argv[0] << " testlist model output" << endl;
		return 1;
	}
	vector<string> ims;
	loadims(argv[1], ims);
	return predict(ims, argv[2], argv[3]);
}
