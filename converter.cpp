#include <fstream>
#include <string>
#include <algorithm>

using namespace std;

int main(int argc, char* argv[]) {
    string inputFile {"debug_output.txt"}, outputFile {"debug_output_c.txt"};
    if (argc == 3) {
        inputFile = argv[1];
        outputFile = argv[2];
    }
    ifstream ifs(inputFile);
    ofstream ofs(outputFile);
    if (!ifs) return -1;
    if (!ofs) return -1;
    double dummyD;
    string dummyS;
    int i {1};
    while (ifs >> dummyD) {
        dummyS = to_string(dummyD);    
        replace(dummyS.begin(), dummyS.end(), '.', ',');
        ofs << dummyS << " ";
        if (i % 4 == 0) {
            ofs << '\n';
        }
        i++;
    }
    ofs.flush();
    ifs.close();
    ofs.close();  
    return 0;
}
