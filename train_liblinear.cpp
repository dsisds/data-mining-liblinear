#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include "xmlreader.h"
#include <boost/filesystem.hpp>
#include "linear.h"

using namespace std;
using namespace cv;

//bool firsttime;
typedef struct libsvm_model_struct {
	struct parameter param;		// set by parse_command_line
	struct problem prob;		// set by read_problem
	struct model *model_;
	struct feature_node* x_space;
}liblinear_model;

typedef struct feature_struct {
    string id;
    vector<int> bbox;
    vector<float> ftr;
} ftrNode;

int parse_postive(string infile, map<string, vector<ftrNode> >& pos){
    ifstream fin(infile.c_str());
    int num, dim;
    fin >> num >> dim;
    for(int i=0;i<num;i++){
        string id;
        vector<int> bbox(4,0);
        vector<float> ftr(dim, 0.0);
        fin >> id;
        if(id==""){
            break;
        }
        fin >> bbox[0] >> bbox[1] >> bbox[2] >> bbox[3];
        for (int j=0;j<dim;j++){
            fin >> ftr[j];
        }
        ftrNode fn;
        fn.id = id;
        fn.bbox = bbox;
        fn.ftr = ftr;
        if(pos.find(id) == pos.end()){
            vector<ftrNode> ft;
            ft.push_back(fn);
            pos.insert(make_pair(id, ft));
        }else{
            pos[id].push_back(fn);
        }
    }
    fin.close();
    return 0;
}

float compute_overlap(vector<int>& win1, vector<int>& win2){
	int xx1=win1[0]>win2[0]?win1[0]:win2[0];
	int yy1=win1[1]>win2[1]?win1[1]:win2[1];
	int xx2=win1[2]<win2[2]?win1[2]:win2[2];
	int yy2=win1[3]<win2[3]?win1[3]:win2[3];
	
	int w=(xx2-xx1+1)>0?(xx2-xx1+1):0;
	int h=(yy2-yy1+1)>0?(yy2-yy1+1):0;

	int inter=w*h;
	int area1 = (win1[2]-win1[0]+1)*(win1[3]-win1[1]+1);
	int area2 = (win2[2]-win2[0]+1)*(win2[3]-win2[1]+1);
	float o = float(inter)/float(area1+area2-inter);

	return o;
}
int is_neg(vector<int>& bbox, map<string, vector< vector<int> > >& gt_boxes, string con, float th){
	if(gt_boxes.find(con)==gt_boxes.end()){
		return 1;
	}
	vector<vector<int> >& gt_box_con = gt_boxes[con];
	for(int i=0;i<gt_box_con.size();i++){
		if(compute_overlap(gt_box_con[i], bbox) > 0.3){
			return 0;
		}
	}
	return 1;
}

double linear_predict(vector<float>& ftr, struct model* m){
	double* w = m->w;
	double bias = m->bias;
	double score = 0;
	assert(ftr.size() == m->nr_feature);
        bias = bias * w[m->nr_feature];
	for(int i=0;i<ftr.size();i++){
		score += w[i]*ftr[i];
	}
	return score+bias;
}
int get_neg(string gtpath, string ftrpath, vector<string>& neg_ims, int index, string con, vector<ftrNode>& neg,liblinear_model* linearmodel,  float thresh, bool *firsttime){
    namespace bf=boost::filesystem;
	//cout << "get_neg:" <<neg.size()<< endl;
    string id = neg_ims[index];
    bf::path ftrfile = ftrpath;
    ftrfile = ftrfile / id;
    ftrfile.replace_extension(".ftr");
    bf::path gtfile = gtpath;
    gtfile = gtfile / id;
    gtfile.replace_extension(".xml");
    map<string, vector< vector<int> > > gt_boxes;
    int status = parsexml(gtfile.string(), gt_boxes);
    ifstream fin(ftrfile.string().c_str());
    int num, dim;
    fin >> num >> dim;
	cout << "id: "<<id <<" num:" <<num << " dim:" << dim << endl;
	if(*firsttime){
		for(int i=0;i<num;i++){
			string im_id;
			fin >> im_id;
			//cout << "processing window(" << (i+1) << "/" << num << ")" << endl;
			if(im_id==""){
				cout <<"reach end" << endl;
				break;
			}else if(im_id != id){
				cout << "box id is invalid. im_id:" << im_id << " id:"<<id << endl;
				return 1;
			}
			vector<int> bbox(4,0);
			vector<float> ftr(dim, 0.0);
			//cout << "reading boxes" << endl;
			fin >> bbox[0] >> bbox[1] >> bbox[2] >> bbox[3];
			//cout << "reading features" << endl;
			for (int j=0;j<dim;j++){
				fin >> ftr[j];
			}
			if(is_neg(bbox, gt_boxes, con, thresh)){
				ftrNode fn;
				fn.id = im_id;
				fn.bbox = bbox;
				fn.ftr = ftr;
				neg.push_back(fn);
			}
		}
		*firsttime = false;
	}else{
		struct model* m;
		if((m=load_model("tmp.model"))==0)
		{
			cout <<"can't open model file ";
			exit(1);
		}
		int nr_class = get_nr_class(m);
		assert(nr_class == 2);
		//double *prob_estimates = (double*)malloc(nr_class*sizeof(double));
		//struct svm_node* x = (struct svm_node *)malloc((dim+1)*sizeof(struct svm_node));
		for(int i=0;i<num;i++){
			string im_id;
			fin >> im_id;
			if(im_id==""){
				break;
			}else if(im_id != id){
				cout << "box id is invalid" << endl;
				//free(x);
				///free(prob_estimates);
				//svm_free_and_destroy_model(&(m));
				return 1;
			}
			vector<int> bbox(4,0);
			vector<float> ftr(dim, 0.0);
			fin >> bbox[0] >> bbox[1] >> bbox[2] >> bbox[3];
			for (int j=0;j<dim;j++){
				fin >> ftr[j];
			}
			//assert(ftrV[i].ftr.size() == ftrdim);
			//vector<float>& ftr_t = ftrV[i].ftr;
			assert(ftr.size() == dim);
			//for(int j=0;j<ftr.size();j++){
			//	x[j].index = j;
			//	x[j].value = double(ftr[j]);
			//}
			//x[ftr.size()].index = -1;
			if(is_neg(bbox, gt_boxes, con, thresh) && linear_predict(ftr, m) > thresh){
				ftrNode fn;
				fn.id = im_id;
				fn.bbox = bbox;
				fn.ftr = ftr;
				neg.push_back(fn);
			}
		}
		//free(x);
		//free(prob_estimates);
		//svm_free_and_destroy_model(&(m));
	}
	//cout << "get_neg done" << endl;
    fin.close();
	//cout << "get_neg done" << endl;
    return 0;
}

int isSameBox(vector<int>& b1, vector<int>& b2){
    assert(b1.size() == b2.size());
    for (int i=0;i<b1.size();i++){
        if(b1[i] != b2[i]){
            return 0;
        }
    }
    return 1;
}

int isDuplicates(map<string, vector<ftrNode> >& train_neg, string id, vector<int>& box){
    if(train_neg.find(id) != train_neg.end()){
        vector<ftrNode>& boxes = train_neg[id];
        for(int i=0;i<boxes.size();i++){
            if(isSameBox(box, boxes[i].bbox)){
                return 1;
            }
        }
    }
    return 0;
}

int merge_hardneg(map<string, vector<ftrNode> >& train_neg, vector<ftrNode>& neg){
    int numAdded = 0;
	cout << "merge hard neg"<<endl;
    for(int i=0;i<neg.size();i++){
        if(!isDuplicates(train_neg, neg[i].id, neg[i].bbox)){
            if(train_neg.find(neg[i].id) != train_neg.end()){
                vector<ftrNode >& t = train_neg[neg[i].id];
                t.push_back(neg[i]);
            }else{
                vector<ftrNode > t;
                t.push_back(neg[i]);
                train_neg.insert(make_pair(neg[i].id, t));
            }
            numAdded++;
        }
    }
    return numAdded;
}

int compute_num(map<string, vector<ftrNode> >& dataSet){
    int ret = 0;
    map<string, vector<ftrNode> >::iterator iter;
    for(iter = dataSet.begin(); iter != dataSet.end(); iter++){
        ret += (iter->second).size();
    }
    return ret;
}


int parseSet_libsvm(map<string, vector<ftrNode> >& negSet, map<string, vector<ftrNode> >& posSet, liblinear_model *linearmodel, int dim, int* num, double bias){
	int num_samples = 0;
	//int dim = 0;
	struct problem *prob = &(linearmodel->prob);
	int posNum, negNum;
	map<string, vector<ftrNode> >::iterator it;
	for(it=negSet.begin();it!=negSet.end();it++){
		num_samples += (it->second).size();
	}
	negNum = num_samples;
	for(it=posSet.begin();it!=posSet.end();it++){
		num_samples += (it->second).size();
	}
	posNum = num_samples - negNum;
	cout << "cache holds " << posNum << " postive samples and " << negNum << " negtive samples" << endl;
	prob->bias = bias;
	prob->l = num_samples;
	prob->n = dim+1;
	prob->y = (double *)malloc(num_samples*sizeof(double));
	prob->x = (struct feature_node **)malloc(num_samples*sizeof(struct feature_node *));
	linearmodel->x_space = (struct feature_node *)malloc((num_samples*(dim+2))*sizeof(struct feature_node));
	int curIdx = 0;
	for(it=posSet.begin();it!=posSet.end();it++){
		vector<ftrNode>& boxes = it->second;
		for(int i=0;i<boxes.size();i++){
			prob->x[curIdx] = &(linearmodel->x_space[curIdx*(dim+2)]);
			prob->y[curIdx] = double(1);
			vector<float>& ftr = boxes[i].ftr;
			assert(ftr.size()==dim);
			for(int j=0;j<dim;j++){
				linearmodel->x_space[curIdx*(dim+2)+j].index = j+1;
				linearmodel->x_space[curIdx*(dim+2)+j].value = double(ftr[j]);
			}
			linearmodel->x_space[curIdx*(dim+2)+dim+1].index = -1;
			linearmodel->x_space[curIdx*(dim+2)+dim].index = dim+1;
			linearmodel->x_space[curIdx*(dim+2)+dim].value = bias;
			curIdx++;
		}
	}
	for(it=negSet.begin();it!=negSet.end();it++){
		vector<ftrNode>& boxes = it->second;
		for(int i=0;i<boxes.size();i++){
			assert(curIdx < prob->l);
			prob->x[curIdx] = &(linearmodel->x_space[curIdx*(dim+2)]);
			prob->y[curIdx] = double(-1.0);
			vector<float>& ftr = boxes[i].ftr;
			assert(ftr.size()==dim);
			for(int j=0;j<dim;j++){
				linearmodel->x_space[curIdx*(dim+2)+j].index = j+1;
				linearmodel->x_space[curIdx*(dim+2)+j].value = double(ftr[j]);
			}
			linearmodel->x_space[curIdx*(dim+2)+dim+1].index = -1;
			linearmodel->x_space[curIdx*(dim+2)+dim].index = dim+1;
			linearmodel->x_space[curIdx*(dim+2)+dim].value = bias;
			curIdx++;
		}
	}
	num[0]= posNum;
	num[1]= negNum;
	/*ofstream fo("tt.train");
	for(int i=0;i<prob->l;i++){
		if(prob->y[i] >0){
		fo << "+" << prob->y[i] << " ";
		}else{
			fo << prob->y[i] << " ";
		}
		for(int j=0;j<dim-1;j++){
			fo<< prob->x[i][j].index<< ":" << prob->x[i][j].value << " ";
		}
		fo << prob->x[i][dim-1].index << ":" << prob->x[i][dim-1].value << endl;
	}
	fo.close();*/
	//cout << "traindata: " << prob->x[331][0].index << ":" <<prob->x[331][0].value <<" "<< prob->x[331][2].index << ":" << prob->x[331][2].value << endl;
	return 0;
}

int free_svm_problem(liblinear_model *linearmodel){
	free(linearmodel->prob.y);
	linearmodel->prob.y = NULL;
	free(linearmodel->prob.x);
	linearmodel->prob.x = NULL;
	free(linearmodel->x_space);
	linearmodel->x_space = NULL;
	return 0;
}


int modify_w(liblinear_model* linearmodel, int* num){
	linearmodel->param.weight[0] = float(num[1])/float(num[0]);
        if(linearmodel->param.weight[0] > 10){
            linearmodel->param.weight[0] = 10;
        }
	//linearmodel->param.weight[0] = 2.0;
	return 0;
}
int train_libsvm(liblinear_model* linearmodel, map<string, vector<ftrNode> >& train_pos, map<string, vector<ftrNode> >& train_neg, int ftrdim, double bias){
	int num_pos_neg[2];
	const char *error_msg;
	parseSet_libsvm(train_neg, train_pos, linearmodel, ftrdim, num_pos_neg, bias);
	//(linearmodel->param).gamma = 1.0 / ftrdim;
	modify_w(linearmodel, num_pos_neg);
	cout << "training libSVM..." << endl;
	error_msg = check_parameter(&(linearmodel->prob),&(linearmodel->param));
	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}
	
	//cout << "traindata: " << model->prob.x[0][0].index << ":" <<model->prob.x[0][0].value <<" "<< model->prob.x[0][2].index << ":" << model->prob.x[0][2].value << endl;
	//cout << "trainLabel:" << model->prob.y[331] << " " << model->prob.y[0] << " gamma:" << model->param.gamma << " C:"<< model->param.C<< endl;
	linearmodel->model_ = train(&(linearmodel->prob), &(linearmodel->param));
	//svm_predict(model->model, model->prob.x[0]);
	//cout << "saving model" << endl;
	save_model("tmp.model",linearmodel->model_);
	//cout << "free model" << endl;
	free_and_destroy_model(&(linearmodel->model_));
	//cout << "free data" << endl;
	free_svm_problem(linearmodel);
	//cout << "train liblinear done" << endl;
	return 0;
}


int shrink_neg_lib(map<string, vector<ftrNode> >& train_neg, liblinear_model* linearmodel, int ftrdim, float threshold){
	//cout << "shrinking neg for libsvm. " << " SV dim:" << ftrdim << endl;
    map<string, vector<ftrNode> > ret;
	map<string, vector<ftrNode> >::iterator saveit;
	//
	struct model* m;
	//svm_save_model("tttt.model",model->model);
	if((m=load_model("tmp.model"))==0)
	{
		cout <<"can't open model file ";
		exit(1);
	}
	//cout << "saving model" << endl;
	//svm_save_model("tttt.model",m);
	int nr_class = get_nr_class(m);
	assert(nr_class == 2);
	//double *prob_estimates = (double*)malloc(nr_class*sizeof(double));
	//struct svm_node* x = (struct svm_node *)malloc((ftrdim+1)*sizeof(struct svm_node));
	int numleft=0;
    for(map<string, vector<ftrNode> >::iterator it=train_neg.begin(); it!=train_neg.end();){
        vector<ftrNode>& ftrV=it->second;
        string id = it->first;
        vector<ftrNode> t;
        for(int i=0;i<ftrV.size();i++){
			assert(ftrV[i].ftr.size() == ftrdim);
			vector<float>& ftr_t = ftrV[i].ftr;
			//cout << "ftr dim:" << ftr_t.size() << endl;
            if(linear_predict(ftr_t, m) > threshold){
                t.push_back(ftrV[i]);
            }
        }
        if(t.size()==0){
			saveit = it;
			it++;
            train_neg.erase(saveit);
        }else{
            train_neg[id] = t;
			numleft+=t.size();
            it++;
        }
    }
	cout << "shrinking neg: cache holds " << numleft << " neg samples" << endl;
    return 0;
}



int data_mining_train_lib(string gtpath, string ftrpath, vector<string>& negims, string pos, string con, int retrain_limit, liblinear_model* linearmodel, float threshold, float hard_thresh,double bias){
    bool first_time = true;
    int max_hard_epoches = 1;
    map<string, vector<ftrNode> > train_pos;
    parse_postive(pos, train_pos);
    int numAdded = 0;
    map<string, vector<ftrNode> > train_neg;
	int dim = -1;
    //initTrainOpts(trainOpts);
	//firsttime=true;
    for (int it=0; it < max_hard_epoches;it++){
        for(int i = 0;i<negims.size();i++){
            cout << "processing neg(" << i+1 << "/" << negims.size() << ")" << endl;
            vector<ftrNode> neg_t;
			int st = get_neg(gtpath, ftrpath, negims, i, con, neg_t, linearmodel, hard_thresh, &first_time);
            if(st){
				cout << "data is wrong" << endl;
                return 1;
            }
			//cout << "get_neg done(main)"<< endl;
			if(neg_t.size() == 0 && i != (negims.size()-1)){
				cout << "No new hard neg" << endl;
				continue;
			}
			int n = 0;
			if(neg_t.size() > 0){
				cout << "getnegNum:" << neg_t.size() << endl;
				if(dim<0){
					dim = neg_t[0].ftr.size();
				}
				//cout << "merge_hardneg(main)"<< endl;
				n = merge_hardneg(train_neg, neg_t);
				numAdded += n;
			}
			cout << "merge_hardneg: added:"<< n << " sum:" << numAdded << " istrainig:" << (numAdded >= retrain_limit || i == (negims.size()-1)) <<endl;
            if(numAdded >= retrain_limit || i == (negims.size()-1)){
                train_libsvm(linearmodel, train_pos, train_neg, dim, bias);
                numAdded = 0;
                shrink_neg_lib(train_neg, linearmodel, dim, threshold);
            }
        }
	}
    return 0;
}

int loadImages(string images, vector<string>& negims){
    ifstream fin(images.c_str());
    while(!fin.eof()){
        string im;
        fin >> im;
        if(im==""){
            return 0;
        }
        negims.push_back(im);
    }
    return 0;
}

int initParam(struct parameter *param){
	param->solver_type = 3;
	param->C = 1e-2;
	param->eps = 0.1; // see setting below
	param->p = 0.1;
	param->nr_weight = 0;
	param->weight_label = NULL;
	param->weight = NULL;
	++(param->nr_weight);
	param->weight_label = (int *)realloc(param->weight_label,sizeof(int)*param->nr_weight);
	param->weight = (double *)realloc(param->weight,sizeof(double)*param->nr_weight);
	param->weight_label[param->nr_weight-1] = 1;
	param->weight[param->nr_weight-1] = 1;
}
int process(int argc, char** argv){
    if(argc<7){
        cout << "Usage:" << argv[0] << " gtpath ftrpath images posfile con model" << endl;
        return 1;
    }
	liblinear_model linearmodel;
    int retrain_limit = 2000;
    float evict_thresh = -1.2;
    float hard_thresh = -1.0001;
	string model_file_name = argv[6];	
    vector<string> negims;
    loadImages(argv[3], negims);
	initParam(&(linearmodel.param));
	//firsttime = true;
    data_mining_train_lib(argv[1], argv[2], negims, argv[4], argv[5], retrain_limit, &linearmodel, evict_thresh, hard_thresh, 10);
	if((linearmodel.model_=load_model("tmp.model"))==0)
	{
		cout <<"can't open model file ";
		exit(1);
	}
	if(save_model(model_file_name.c_str(),linearmodel.model_)){
			fprintf(stderr, "can't save model to file %s\n", model_file_name.c_str());
			exit(1);
	}
	free_and_destroy_model(&(linearmodel.model_));
	destroy_param(&(linearmodel.param));
    return 0;
}

int main(int argc, char** argv){
        return process(argc, argv);
}
