#include <stdio.h>
#include <tensorflow/c/c_api.h>

#include <iostream>

void DeallocateBuffer(void* data, size_t);
TF_Buffer* ReadBufferFromFile(const char* file);


int main(int argc, char* argv[]) {
    if (argc != 3)
    {
        std::cerr << std::endl << "Usage: ./project path_to_graph.pb path_to_image.png" << std::endl;
        return 1;
    }

    std::string graph_path = argv[1];

    TF_Buffer* buffer = ReadBufferFromFile(graph_path.c_str());
    if (buffer == nullptr) {
        printf("Can't read buffer from file\n");
        return 1;
    }

    TF_Graph* graph = TF_NewGraph();
    TF_Status* status = TF_NewStatus();
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();

    TF_GraphImportGraphDef(graph, buffer, opts, status);
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(buffer);

    if (TF_GetCode(status) != TF_OK) {
        TF_DeleteStatus(status);
        TF_DeleteGraph(graph);
        printf("Can't import GraphDef\n");
        return 2;
    }

    printf("Load draph success\n");

    TF_DeleteStatus(status);
    TF_DeleteGraph(graph);

    return 0;
}

void DeallocateBuffer(void* data, size_t) {
    std::free(data);
}

TF_Buffer* ReadBufferFromFile(const char* file) {
    const auto f = std::fopen(file, "rb");
    if (f == nullptr) {
        return nullptr;
    }

    std::fseek(f, 0, SEEK_END);
    const auto fsize = ftell(f);
    std::fseek(f, 0, SEEK_SET);

    if (fsize < 1) {
        std::fclose(f);
        return nullptr;
    }

    const auto data = std::malloc(fsize);
    std::fread(data, fsize, 1, f);
    std::fclose(f);

    TF_Buffer* buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = DeallocateBuffer;

    return buf;
}
