/* Inspect an authenticated vendor-extracted KL720 setup with the installed PLUS headers. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kneron_setup_reader.h"

static int print_node(const char *role, KneronSetup_Tensor_vec_t nodes) {
    if (KneronSetup_Tensor_vec_len(nodes) != 1) return 2;
    KneronSetup_Tensor_table_t node = KneronSetup_Tensor_vec_at(nodes, 0);
    KneronSetup_QuantizationParameters_table_t q = KneronSetup_Tensor_quantization(node);
    if (q == NULL) return 2;
    KneronSetup_FxpInfo_vec_t params = KneronSetup_QuantizationParameters_fxp_info(q);
    for (size_t i = 0; i < KneronSetup_FxpInfo_vec_len(params); ++i) {
        KneronSetup_FxpInfo_table_t p = KneronSetup_FxpInfo_vec_at(params, i);
        float scale = KneronSetup_FxpInfo_scale(p);
        uint32_t bits;
        memcpy(&bits, &scale, sizeof(bits));
        printf("%s\t%s\t%u\t%zu\t%08x\t%d\n", role, KneronSetup_Tensor_name(node),
               KneronSetup_Tensor_format(node), i, bits, KneronSetup_FxpInfo_radix(p));
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f || fseek(f, 0, SEEK_END)) return 2;
    long len = ftell(f);
    if (len < 32 || len > 1048576 || fseek(f, 0, SEEK_SET)) return 2;
    unsigned char *data = malloc((size_t)len);
    if (!data || fread(data, 1, (size_t)len, f) != (size_t)len) return 2;
    fclose(f);
    KneronSetup_INFContent_table_t model = KneronSetup_INFContent_as_root(data);
    if (!model || KneronSetup_INFContent_magic(model) != 0x012EBD34) return 2;
    int result = print_node("input", KneronSetup_INFContent_inputs(model));
    if (!result) result = print_node("output", KneronSetup_INFContent_outputs(model));
    free(data);
    return result;
}
