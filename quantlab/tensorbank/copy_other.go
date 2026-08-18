//go:build !linux

package tensorbank

import "fmt"

func copyFileRangeAll(rfd, wfd int, roff, woff, remain int64) error {
	return fmt.Errorf("copy_file_range: not supported")
}

func adviseSequential(fd int, off, length int64) {}

// DiskFree cannot query free space on this platform.
func DiskFree(path string) (uint64, bool) {
	_ = path
	return 0, false
}
