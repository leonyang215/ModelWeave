// Package plugins installs the compiled-in, repository-private scorer directory.
// Add a blank import here (or in a custom gateway binary) to enable a plugin;
// registration never requires changing the scheduler's selection loop.
package plugins

import (
	_ "github.com/leonyang215/ModelWeave/internal/scheduler/plugins/affinity"
	_ "github.com/leonyang215/ModelWeave/internal/scheduler/plugins/label"
	_ "github.com/leonyang215/ModelWeave/internal/scheduler/plugins/load"
	_ "github.com/leonyang215/ModelWeave/internal/scheduler/plugins/prefixcache"
)
