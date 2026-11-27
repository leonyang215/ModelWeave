// Package scheduling installs the compiled-in scoring plugin directory.
// Add a blank import here (or in a custom gateway binary) to enable a plugin;
// registration never requires changing the scheduler's selection loop.
package scheduling

import (
	_ "github.com/leonyang215/ModelWeave/plugins/scheduling/affinity"
	_ "github.com/leonyang215/ModelWeave/plugins/scheduling/label"
	_ "github.com/leonyang215/ModelWeave/plugins/scheduling/load"
	_ "github.com/leonyang215/ModelWeave/plugins/scheduling/prefixcache"
)
