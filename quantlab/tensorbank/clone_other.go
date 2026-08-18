//go:build !linux

package tensorbank

import "fmt"

func cloneFileFast(srcPath, dstPath string) error {
	_ = srcPath
	_ = dstPath
	return fmt.Errorf("tensorbank: no filesystem clone")
}
