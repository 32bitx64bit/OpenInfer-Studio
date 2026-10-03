package mediagen

import (
	"reflect"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

func TestSDMemoryHeadroomHonorsCapabilitiesAndUserBudgets(t *testing.T) {
	for _, tc := range []struct {
		name, backend, help string
		args                []string
		reserve             bool
	}{
		{"automatic HIP", runtimes.BackendHIP, "--max-vram", []string{"--diffusion-model", "ltx.gguf"}, true},
		{"explicit cap", runtimes.BackendHIP, "--max-vram", []string{"--max-vram", "12"}, false},
		{"explicit opt out", runtimes.BackendHIP, "--max-vram", []string{"--max-vram", "0"}, false},
		{"explicit reserve", runtimes.BackendHIP, "--max-vram", []string{"--max-vram", "-3"}, false},
		{"unsupported runtime", runtimes.BackendHIP, "--model", nil, false},
		{"CPU computation", runtimes.BackendHIP, "--max-vram --backend", []string{"--backend", "cpu"}, false},
		{"other backend", runtimes.BackendCUDA, "--max-vram", nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			original := append([]string(nil), tc.args...)
			got, warning := applySDMemoryHeadroom(tc.args, tc.backend, ParseSDCapabilities(tc.help), tc.help)
			if tc.reserve {
				want := append(original, "--max-vram", "-2")
				if !reflect.DeepEqual(got, want) || warning == "" {
					t.Fatalf("automatic reserve = %v %q", got, warning)
				}
			} else if !reflect.DeepEqual(got, original) || warning != "" {
				t.Fatalf("explicit or unsupported configuration changed: %v %q", got, warning)
			}
		})
	}
}
