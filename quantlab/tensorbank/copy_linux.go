//go:build linux

package tensorbank

import (
	"fmt"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// copyFileRangeMax is Linux's documented per-call cap (INT_MAX & ~0xFFF).
const copyFileRangeMax = 0x7ffff000

// copyFileRangeAll copies remain bytes from rfd at roff to wfd at woff via
// unix.CopyFileRange. Offsets are updated as the copy proceeds.
func copyFileRangeAll(rfd, wfd int, roff, woff, remain int64) error {
	for remain > 0 {
		chunk := remain
		if chunk > copyFileRangeMax {
			chunk = copyFileRangeMax
		}
		n, err := unix.CopyFileRange(rfd, &roff, wfd, &woff, int(chunk), 0)
		if err != nil {
			return err
		}
		if n == 0 {
			return fmt.Errorf("copy_file_range: zero-length copy")
		}
		remain -= int64(n)
	}
	return nil
}

func adviseSequential(fd int, off, length int64) {
	if length <= 0 {
		return
	}
	_ = unix.Fadvise(fd, off, length, unix.FADV_SEQUENTIAL)
}

// DiskFree reports available bytes on the filesystem containing path.
// ok is false when the query fails.
func DiskFree(path string) (uint64, bool) {
	var st unix.Statfs_t
	dir := path
	for {
		if err := unix.Statfs(dir, &st); err == nil {
			bsize := uint64(st.Bsize)
			if bsize == 0 {
				return 0, false
			}
			return uint64(st.Bavail) * bsize, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return 0, false
		}
		dir = parent
	}
}
