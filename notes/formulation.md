
- partition a device: 
	- slab (1d cut along transport), 
	- tile (3d cut), tile can be nested under slab
- approximating density matrix (DM): 
	- local (take a partition of A_PP, embed it in an environment A_EE, then get G_PP), 
	- global (take the whole system, embed it in the true OBC then get G). Of cause we can use RGF with slab partition to reduce cost.
	- DM is G integrated over a complex contour. 
- with DM, we apply randomized range finder (QR) and get columns with the dominant lambda, form an orthonormal basis U to project the system into a mode space (MS)
- with a full DM, we can take its diagonal blocks, and define basis in these sub-partitions, they are orthonormal by construction, but they might not span the same space as the basis obtained from the full DM. This is a locality cost, since we enforce locality into the basis. 
- we search the solution of Ax=b in the MS with Ac = U' A U and bc = U' b and xc = U' x


Two open questions:

1. Can we find a block-diagonal basis (we can accept it to be a little bigger) that contains the vector space spanned by a dense matrix basis? (like Wannier transformation)

2. Can we approximate density matrix and still get a basis that solves accurately the original equation?






locally one could do shift inverse? put this shift at the middle energy or somewhere below to get the "realest" modes first. Maybe the modes can be first found everywhere only with CAP, then for the contacts we could refine or something like Woodbury

since being linear is nicer

showing nonlinear eigenstates, give the reasoning why the true U is a good preconditioner, why due to the locality of U U1 should be good, why schwarz is added.


showing residual/transmission error drop curve


makes a strong case