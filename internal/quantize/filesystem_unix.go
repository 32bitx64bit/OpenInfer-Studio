//go:build !windows

package quantize

import "golang.org/x/sys/unix"

func sameFilesystem(a, b string) (bool, bool) {
	var sa, sb unix.Stat_t
	if err := unix.Stat(a, &sa); err != nil {
		return false, false
	}
	if err := unix.Stat(b, &sb); err != nil {
		return false, false
	}
	return uint64(sa.Dev) == uint64(sb.Dev), true
}
