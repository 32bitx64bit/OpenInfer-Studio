//go:build windows

package quantize

import (
	"path/filepath"
	"strings"
)

func sameFilesystem(a, b string) (bool, bool) {
	aa, err := filepath.Abs(a)
	if err != nil {
		return false, false
	}
	bb, err := filepath.Abs(b)
	if err != nil {
		return false, false
	}
	va, vb := filepath.VolumeName(aa), filepath.VolumeName(bb)
	if va == "" || vb == "" {
		return false, false
	}
	return strings.EqualFold(va, vb), true
}
