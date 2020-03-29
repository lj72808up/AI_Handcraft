#include <iostream>
#include <fstream>
#include <string>
#include<vector>
#include <stdio.h>
#include <string.h>
using namespace std;

int matrixMultiply(){
    int matrix_a[2][3]={{1,2,3},{4,5,6}};//a矩阵2X3
    int matrix_b[3][4]={{1,2,3,4},{5,6,7,8},{1,2,3,4}};//b矩阵3X4
    int matrix_result[2][3];//结果矩阵2X4


    for(int m=0;m<2;m++){
        for(int s=0;s<3;s++){
            matrix_result[m][s]=0;//变量使用前记得初始化,否则结果具有不确定性
            for(int n=0;n<3;n++){
                matrix_result[m][s]+=matrix_a[m][n]*matrix_b[n][s];
            }
        }
    }
    for(int m=0;m<2;m++){
        for(int s=0;s<3;s++){
            cout<<matrix_result[m][s]<<"\t";
        }
        cout<<endl;
    }

    return 0;
}

int testSplit(char s1[], char *sep1){
    char s[] = "a,b,c,d";
    const char *sep = ",*"; //可按多个字符来分割
    char *p;
    p = strtok(s, sep);
    while(p){
        printf("%s ", p);
        p = strtok(NULL, sep);
    }
    printf("\n");
    return 0;
}

int main(int argc, char*argv[]) {
//    testSplit((char*)"a,b,c,d",(char*)",");
    matrixMultiply();
}
int test(){
    ifstream read_file;
    read_file.open(R"(D:\clion\Hoomework\matrix.txt)", ios::binary);

    string line;
    int lineNumber = 0;  // 当前行数
    int matrixSize = 0;  // 矩阵大小
    int index1 = 0;
    int index2 = 0;  // 2个矩阵读入的行数
    vector<int*> data1, data2;  // 2个二维数组


    while(getline(read_file, line)){
        if (lineNumber==0){
            matrixSize = std::stoi(line.c_str());
            cout << "matrix size:" << matrixSize << endl;
        }else {
            if (index1 < matrixSize){  // 矩阵1的数据
                index1 ++;
            }else if(index2 < matrixSize){
                index2 ++;
            }else {
                break;
            }
            cout<<"line:"<<line.c_str()<<endl;
        }
       lineNumber++;
    }

    return 0;
}


