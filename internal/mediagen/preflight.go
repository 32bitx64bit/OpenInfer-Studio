package mediagen

import (
	"fmt"
	"path/filepath"

	"github.com/openinfer/openinfer-studio/internal/sdmodel"
)

func vaeIncompatibility(path string) string {
	if path == "" {
		return ""
	}
	names, _ := sdmodel.TensorNames(path)
	return sdmodel.VAEIncompatibility(path, names)
}

// validateLaunchComponents examines the final argv, including raw overrides,
// before any restoration work or process creation. Check the effective (last)
// --vae, as sd-server does, so an explicit compatible override can repair an
// automatically discovered incompatible companion.
func validateLaunchComponents(args []string) error {
	var vae string
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "--vae" {
			vae = args[i+1]
		}
	}
	if why := vaeIncompatibility(vae); why != "" {
		return fmt.Errorf("incompatible VAE %s: %s", filepath.Base(vae), why)
	}
	return nil
}
